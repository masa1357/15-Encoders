"""
Transformer Encoder のアーキテクチャを1から実装する

"""

# INFO: importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import math, copy, re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns

# import torchtext
import matplotlib.pyplot as plt
from typing import Optional


warnings.simplefilter("ignore")
print(torch.__version__)

# ここからのimportは仮
from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# activationsのパスを変更

from transformers.activations import ACT2FN, gelu
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

roberta_path = (
    "/usr/local/lib/python3.10/dist-packages/transformers/models/roberta/__init__.py"
)
from transformers.models.roberta.configuration_roberta import RobertaConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]

#INFO: 既存のBERTモデルのコードをコピー: from transformers.models.bert.modeling_bert.BertEmbeddings
#INFO: クラス構造図
# PreTrainedModel                       # -> 外部インポート
# └ RobertaPreTrainedModel              # -> 実装済
#    └ RobertaModel                     # -> 実装済
#       └ RobertaEmbeddings             # -> 実装済
#       └ RobertaEncoder                # -> 実装済
#           └ RobertaLayer              # -> 実装済
#               └ RobertaAttention      #! -> 未実装
#               └ RobertaSelfAttention  # -> 実装済
#               └ RobertaIntermediate   #! -> 未実装
#               └ RobertaSelfOutput     # -> 実装済
#               └ RobertaOutput         #! -> 未実装
#       └ RobertaPooler                 # -> 実装済
#    └ RobertaFor15LayersClassification # -> 実装済
#       └ RobertaModel(instance)        # -> 実装済
#       └ RobertaClassificationHead     # -> 実装済

class RobertaEmbeddings(nn.Module): # -> 実装済
    """
    トークンに対して位置情報を付加するための埋め込み層
    BERTの埋め込みと同様の実装とほぼ同じだが，位置埋め込みのindexに若干の変更がある
    """

    #INFO: 既存のBERTモデルのコードをコピー: from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        #INFO: 単語埋め込み層の定義
        #INFO: input_tokens -> word_embeddings -> position_embeddings -> token_type_embeddings -> embeddings
        # -> input_tokens : トークン化された単語やサブワードの ID（整数値）。これらはモデル内部で埋め込み層を通してベクトル表現に変換される。
        # -> word_embeddings : 単語埋め込み層。入力トークン ID に対応する埋め込みベクトルを返す。
        # -> position_embeddings : 位置埋め込み層。入力トークンの位置に対応する埋め込みベクトルを返す。
        # -> token_type_embeddings : トークンタイプ埋め込み層。入力トークンのセグメント ID に対応する埋め込みベクトルを返す。
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        #INFO: ここからの定義はinputに対してのEmbedding内で適用するMLPに関する
        #INFO: LayerNormalization の定義
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        #INFO: Dropout の定義
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized

        #INFO: 位置埋め込みのタイプの設定
        #INFO: 位置埋め込みのタイプは，絶対位置 or 相対位置
        #INFO: gettattr() はオブジェクトの属性を取得する
        # -> 絶対位置埋め込み：位置IDをそのまま埋め込む
        # -> 相対位置埋め込み：位置IDを相対位置に変換して埋め込む
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        #INFO: 位置IDを保持するバッファの確保
        #INFO: torch.arange -> 0 から config.max_position_embeddings - 1 までの連続した整数を生成
        #INFO: expand((1, -1)) -> テンソルを拡張,新しい次元を追加
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        #INFO: トークンタイプIDを保持するバッファの確保
        #INFO: トークンタイプIDは，セグメントIDを示す
        #INFO: 0で初期化
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        """
        補足
        register_buffer について :
            モデルの属性としてテンソルを登録しますが、学習パラメータ（nn.Parameter）としては扱われません。
            -> バッファはオプティマイザの更新対象にはなりませんが、モデルと共に保存・ロードされます（persistent=False の場合を除く）。

            バッファの登録：
                self.token_type_ids や self.position_ids はバッファとして登録されています。
            メリット：
                モデルと共に保存・ロードされます。
                学習パラメータではないため、勾配計算やパラメータ更新の対象外です。
            persistent=False :
                バッファをモデルの状態に含めず、保存・ロード時に無視します。
            理由：
                これらのバッファは固定の値であり、モデルの再現性に影響を与えないため。
                モデルサイズを小さく保つことができます。
        
        persistent=False の効果 :
            モデルの保存・ロード時にこのバッファを含めません。
            一時的なデータや計算に必要な値を保持する場合に使用します。
        
        パディングトークンの扱い :
            埋め込み層で padding_idx を指定すると、そのインデックスに対応する埋め込みベクトルは常にゼロになります。
            -> パディング部分がモデルの計算に影響を与えないようにします。
        

        Layer Normalization の命名 :
            self.LayerNorm が CamelCase で命名されているのは、TensorFlow の変数名と一致させるためです。
            -> TensorFlow で事前学習されたモデルの重みを PyTorch モデルにロードする際に、名前の不一致による問題を防ぎます。
        """

        #INFO: padding_idx の設定 -> configの値
        self.padding_idx = config.pad_token_id
        #INFO: Embeddingの定義
        #INFO: ここで整数値だったinputをベクトル表現（意味表現，埋め込み）に変換
        # -> config.max_position_embeddings   : Embeddingの語彙数（最大位置数）
        # -> config.hidden_size               : 埋め込みベクトルの次元数
        # -> padding_idx                      : このindexに対応する埋め込みベクトルは0になる
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        #INFO: input_ids が None の場合，inputs_embeds が None でないことを確認
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        """
        input_ids と inputs_embeds
            input_ids：
                トークン化された単語やサブワードの ID（整数値）のテンソル。
                モデル内部で埋め込み層によってベクトルに変換されます。
            inputs_embeds：
                既に埋め込みベクトルに変換された入力。
                ユーザーが独自に埋め込みを計算してモデルに渡す場合に使用します。
            相互排他的：
                input_ids と inputs_embeds の両方を同時に指定することはできません。
                どちらか一方を指定する必要があります。
        """

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        #INFO: sequence_length の取得
        seq_length = input_shape[1]


        #INFO: token_type_ids が None の場合，0で初期化
        #INFO: token_type_ids が None でない場合，buffered_token_type_ids に token_type_ids を格納
        # -> ユーザーが token_type_ids を渡さずにモデルをトレースする際に役立ち、Issue #5664 を解決する
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                #INFO: self.token_type_ids から、必要な長さ（seq_length）だけを切り取ります。形状：(1, seq_length)
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                #INFO: batch size(input_shape[0]) に合わせてテンソルを拡張
                #! expand の注意点：メモリ効率のため、実際のデータコピーは行わず、ビューを作成します。
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        """
        token_type_ids の役割
            トークンタイプ埋め込みは、トークンがどのセグメント（文）に属するかをモデルに知らせるために使用されます。

            例：
            文のペア（質問と回答など）を入力する場合、最初の文のトークンには 0、二番目の文のトークンには 1 を割り当てます。   
        RoBERTa モデル：
            RoBERTa はトークンタイプ埋め込みを使用しない設計になっています。
            しかし、コードの汎用性や互換性を保つために token_type_ids を処理しています。
            全てゼロのトークンタイプ埋め込みを使用します。     
        """

        #INFO: 単語埋め込みの取得
        #INFO: inputs_embeds が指定されていない場合の処理
        if inputs_embeds is None:
            #INFO: input_ids を埋め込みベクトルに変換
            inputs_embeds = self.word_embeddings(input_ids)
        
        #INFO: トークンタイプ埋め込みの取得
        #INFO: token_type_embeddings（形状：(batch_size, seq_length, hidden_size)）
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        #INFO: 埋め込みの合計
        #INFO: embeddings（形状：(batch_size, seq_length, hidden_size)）
        #INFO: 1.単語埋め込みとトークンタイプ埋め込みを要素ごとに加算
        embeddings = inputs_embeds + token_type_embeddings
        #INFO: 2.位置埋め込みを加算（絶対位置埋め込みの場合）
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        #INFO: 3.LayerNormalization を適用
        embeddings = self.LayerNorm(embeddings)
        #INFO: 4.Dropout を適用
        embeddings = self.dropout(embeddings)

        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class RobertaSelfAttention(nn.Module): # -> 実装済
    """
    RoBERTa の Self-Attention 実装
    入力テンソルに対して Self-Attention を適用し、情報を集約
    マルチヘッドアテンションを使用し、複数の異なる表現を学習
    """
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        #INFO: hidden_size が num_attention_heads の整数倍か確認
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            #INFO: False の場合，エラー
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        #INFO: Attention head のサイズ計算
        # -> hidden_size / num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        #INFO: 本質的には hidden_size と同じ
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        #INFO: Query, Key, Value の線形変換層の定義
        # -> 学習可能パラメータ
        #INFO: hidden_size -> all_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        #INFO: Attention ドロップアウトの定義
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        #INFO: 位置埋め込みのタイプの設定
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        #INFO: 相対位置埋め込みの場合，距離埋め込みの定義
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        #INFO: デコーダモデルの設定を取得
        self.is_decoder = config.is_decoder

    #INFO: スコア計算のためのテンソル変換関数
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor    : 入力テンソル
            size            : (batch_size, sequence_length, all_head_size)
        
        Returns
        -------
        torch.Tensor        : 転置されたテンソル
            size            : (batch_size, num_attention_heads, sequence_length, attention_head_size)
        """
        #INFO: テンソルの最後の次元を(num_attention_heads, attention_head_size)に変換
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        #INFO: view() でテンソルの形状を変更
        # -> (batch_size, sequence_length, all_head_size)
        # -> (batch_size, sequence_length, num_attention_heads, attention_head_size)
        x = x.view(new_x_shape)
        #INFO: permute() で次元の順番を並び替える
        # -> (batch_size, num_attention_heads, sequence_length, attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states           : torch.Tensor,
        attention_mask          : Optional[torch.FloatTensor] = None,
        head_mask               : Optional[torch.FloatTensor] = None,
        encoder_hidden_states   : Optional[torch.FloatTensor] = None,
        encoder_attention_mask  : Optional[torch.FloatTensor] = None,
        past_key_value          : Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions       : Optional[bool] = False,
    )   -> Tuple[torch.Tensor]:
    """
    このメソッドは、自己注意（Self-Attention）やクロスアテンション（Cross-Attention）を計算し、隠れ状態を更新します。

    Parameters
    ----------
    hidden_states           : torch.Tensor
        現在のレイヤーへの入力隠れ状態。形状は `(batch_size, sequence_length, hidden_size)`。

    attention_mask          : Optional[torch.FloatTensor], オプション
        アテンションマスク。形状は `(batch_size, 1, 1, sequence_length)`。

    head_mask               : Optional[torch.FloatTensor], オプション
        アテンションヘッドをマスクするためのテンソル。

    encoder_hidden_states   : Optional[torch.FloatTensor], オプション
        エンコーダの隠れ状態。クロスアテンションで使用。

    encoder_attention_mask  : Optional[torch.FloatTensor], オプション
        エンコーダのアテンションマスク。

    past_key_value          : Optional[Tuple[Tuple[torch.FloatTensor]]], オプション
        過去のキーとバリューのキャッシュ。

    output_attentions       : Optional[bool], オプション
        アテンションの重みを出力するかどうか。

    Returns
    -------
    Tuple[torch.Tensor]
        更新された隠れ状態やアテンションの重み、キャッシュを含むタプル。

    """
        #INFO: queryの計算
        # -> hidden_states に線形変換を適用し，queryを求める
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        #INFO: encoder_hidden_states が提供されている場合はcross-attentionを実行
        is_cross_attention = encoder_hidden_states is not None

        #INFO: cross-attentionの計算部分
        #INFO: case1 : cross-attention かつ過去のキャッシュがある場合
        if is_cross_attention and past_key_value is not None:
            #INFO: 過去のキャッシュを再利用して，keyとvalueを取得
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        #INFO: case2 : cross-attention かつ過去のキャッシュがない場合
        elif is_cross_attention:
            #INFO: encoder_hidden_states に線形変換を適用し，keyとvalueを求める
            #INFO: transpose_for_scores() でテンソルの形状を変更
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        #INFO: case3 : cross-attention でない場合 かつ 過去のキャッシュがある場合
        elif past_key_value is not None:
            #INFO: 過去のキャッシュを再利用して，keyとvalueを取得
            # -> cross-attention 出ない場合は，引数がhidden_states となる
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        #INFO: case4 : cross-attention でない場合 かつ 過去のキャッシュがない場合
        else:
            #INFO: hidden_states（現在の隠れ状態） に線形変換を適用し，keyとvalueを求める
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        #INFO: query_layer の形状を変更
        query_layer = self.transpose_for_scores(mixed_query_layer)

        #INFO: 過去のキャッシュがあるかどうか
        use_cache = past_key_value is not None
        #INFO: decoder の場合，keyとvalueをキャッシュ
        if self.is_decoder:
            # クロスアテンションの場合は、すべてのクロスアテンションのキー/値の状態の Tuple(torch.Tensor, torch.Tensor) を保存します。
            # クロスアテンション レイヤーへの以降の呼び出しでは、すべてのクロスアテンションのキー/値の状態を再利用できます (最初の "if" ケース)
            # 単方向セルフアテンション (デコーダー) の場合は、以前のすべてのデコーダーのキー/値の状態の Tuple(torch.Tensor, torch.Tensor) を保存します。単方向セルフアテンションへの以降の呼び出しでは、以前のデコーダーのキー/値の状態を現在の投影されたキー/値の状態と連結できます (3 番目の "elif" ケース)
            # エンコーダーの双方向セルフアテンションの場合は、`past_key_value` は常に `None` です
            past_key_value = (key_layer, value_layer)

        #INFO: Attention スコアの計算
        #INFO: query_layer と key_layer の内積を取ることで，Attention スコアを計算
        #INFO: 形状変換も行う
        # -> (batch_size, num_heads, query_length, key_length)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        #INFO: 相対位置埋め込みの計算・適用
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            #INFO: 位置IDの計算
            #INFO: use_cache が True の場合，key_length - 1 を使用して計算
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            #INFO: use_cache が False の場合，query_length を使用して計算
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            
            #INFO: key_length を計算
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            #INFO: 位置IDの差分を計算
            # -> l: query_length, r: key_length
            distance = position_ids_l - position_ids_r

            #INFO: 距離埋め込みの取得
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            #INFO: queryのデータ型に合わせる
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            #INFO: 相対位置埋め込みの場合，query_layer に距離埋め込みを加算
            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        #INFO: Attention スケーリング
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #INFO: Attention マスクの適用
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        #INFO: Attention スコアの正規化
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        #INFO: Attention ドロップアウトの適用
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        #INFO: head_mask が指定されている場合，マスクを適用してアテンションヘッドを無効化
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        #INFO: value_layer に attention_probs(確率値) を適用して，context_layer（新規隠れ状態） を計算
        context_layer = torch.matmul(attention_probs, value_layer)

        #INFO: context_layer の形状を変更
        # -> (batch_size, sequence_length, hidden_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        #INFO: 出力の設定
        # -> context_layer : 新規隠れ状態
        # -> attention_probs : アテンションの重み
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        #INFO: Decoder モデルの場合，キャッシュを返す
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class RobertaSelfOutput(nn.Module): # -> 実装済
    """
    Self-Attention の出力に対して，残差接続と LayerNormalization を適用
    """
    def __init__(self, config):
        super().__init__()
        #INFO: 線形変換層(dense)の定義
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #INFO: LayerNormalization の定義
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        #INFO: Dropout の定義
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor
            Self-Attention の出力
        input_tensor : torch.Tensor
            入力テンソル
        
        Returns
        -------
        torch.Tensor
            出力テンソル
        """
        #INFO: 各層の処理を順次適用
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        """
        補足情報
        残差接続（Residual Connection）について
            目的：
            深いニューラルネットワークでの勾配消失問題を緩和します。
            方法：
            レイヤーの入力をその出力に直接足し合わせます。
            効果：
            ネットワークの学習が容易になり、より深い構造のモデルが訓練可能になります。
        Layer Normalization について
            目的：
            各サンプルの特定の層内での出力を正規化し、学習を安定化させます。
            方法：
            入力テンソルの各タイムステップ（シーケンスの各位置）ごとに、特徴次元に沿って平均と分散を計算し、正規化します。
            効果：
            内部共変量シフトを軽減し、学習速度の向上や収束性の改善につながります。
        ドロップアウトについて
            目的：
            過学習を防ぐために、学習時にランダムにノードを無効化します。
            方法：
            指定した確率でノードの出力をゼロにします。
            効果：
            モデルがより一般的な特徴を学習し、汎化性能が向上します。
        """
        return hidden_states

class RobertaAttention(nn.Module): #! 未実装

class RobertaIntermediate(nn.Module): #! 未実装

class RobertaOutput(nn.Module): #! 未実装


#INFO: 既存のBERTモデルのコードをコピー: from transformers.models.bert.modeling_bert.BertLayer with Bert->Roberta
class RobertaLayer(nn.Module):
    """
    Encoder の内部構造を定義
    -> Encoder Block
        -> Attention -> Intermediate -> Output
    """

    #INFO: 初回実行
    def __init__(self, config):
        super().__init__()

        #INFO: FeedForward ネットワークのチャンクサイズを設定
        # -> チャンクサイズを設定することで，大きなテンソルを小さなテンソルに分割して順次処理することができる
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        #INFO: シーケンス長の次元を設定
        self.seq_len_dim = 1
        #INFO: Self-Attention レイヤーの定義
        self.attention = RobertaAttention(config)
        #INFO: Decoder モデルの場合，Cross-Attention レイヤーを追加
        #INFO: is_decoder : デコーダフラグ
        #INFO: add_cross_attention : クロスアテンションフラグ
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            #INFO: Decoderと明示していないならエラー
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            #INFO: Cross-Attention レイヤーの定義
            # -> 絶対位置埋め込みを使用
            self.crossattention = RobertaAttention(config, position_embedding_type="absolute")

        #INFO: FeedForward ネットワークの定義
        #INFO: 中間層の定義
        #INFO: 活性化関数などの処理を行う
        #! ここまとめられないかな？
        self.intermediate = RobertaIntermediate(config)
        #INFO: 出力層の定義
        #INFO: 残差結合やLayerNormalizationを行う
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states           : torch.Tensor,
        attention_mask          : Optional[torch.FloatTensor]               = None,
        head_mask               : Optional[torch.FloatTensor]               = None,
        encoder_hidden_states   : Optional[torch.FloatTensor]               = None,
        encoder_attention_mask  : Optional[torch.FloatTensor]               = None,
        past_key_value          : Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions       : Optional[bool]                            = False,
    ) -> Tuple[torch.Tensor]:
        """
        RobertaLayer を通じて順伝播（フォワードパス）を実行

        Parameters
        ----------
        hidden_states : torch.Tensor
            前の層からの隠れ状態を表す入力テンソル。形状は `(batch_size, sequence_length, hidden_size)` です。

        attention_mask : Optional[torch.FloatTensor], オプション
            アテンションマスクのテンソル。形状は `(batch_size, 1, 1, sequence_length)` で、デフォルトは `None` です。
            アテンション機構で特定の位置（例：パディングトークン）への注意を防ぐために使用されます。
            マスクの値は 0（マスクする）または 1（マスクしない）です。

        head_mask : Optional[torch.FloatTensor], オプション
            ヘッドマスクのテンソル。形状は `(num_heads,)` または `(num_layers, num_heads)` で、デフォルトは `None` です。
            自己注意機構内で特定のアテンションヘッドをマスクするために使用されます。
            マスクの値は 0（マスクする）または 1（マスクしない）です。

        encoder_hidden_states : Optional[torch.FloatTensor], オプション
            エンコーダの隠れ状態のテンソル。形状は `(batch_size, encoder_sequence_length, hidden_size)` で、デフォルトは `None` です。
            モデルが `add_cross_attention=True` でデコーダとして設定されている場合、クロスアテンションで使用されます。

        encoder_attention_mask : Optional[torch.FloatTensor], オプション
            エンコーダのアテンションマスクのテンソル。形状は `(batch_size, 1, 1, encoder_sequence_length)` で、デフォルトは `None` です。
            クロスアテンション中にエンコーダの入力内の特定の位置をマスクするために使用されます。
            `encoder_hidden_states` が提供されている場合にのみ関連します。

        past_key_value : Optional[Tuple[Tuple[torch.FloatTensor]]], オプション
            前のステップで計算されたキーとバリューのテンソルを含むタプル。デフォルトは `None` です。
            デコーダーモデルでの逐次デコードを高速化するために使用されます。
            各タプルには2つのテンソルが含まれます：形状が `(batch_size, num_heads, sequence_length, head_dim)` のキーとバリューのテンソルです。

        output_attentions : Optional[bool], オプション
            `True` に設定すると、出力の一部としてアテンションの重みが返されます。デフォルトは `False` です。

        Returns
        -------
        Tuple[torch.Tensor]
            以下を含むタプル：

            - **hidden_states** (`torch.Tensor`): レイヤーの出力隠れ状態。形状は `(batch_size, sequence_length, hidden_size)`。

            - **present_key_value** (`Tuple[Tuple[torch.FloatTensor]]`, オプション): 更新された過去のキーとバリューのテンソル。
            `past_key_value` が提供されている場合に返されます（デコーダーモデルでのみ関連）。

            - **attentions** (`torch.FloatTensor`, オプション): 自己アテンション（および適用可能な場合はクロスアテンション）のアテンション重み。
            `output_attentions` が `True` の場合に返されます。
        """

        #INFO: 自己注意機構（Self-Attention）で使用する過去のキーとバリュー（past_key_value）を取得
        # -> 推論の高速化
        #INFO: pat_key_value は過去のキーとバリューの状態を含むタプル -> [0]:key, [1]:value
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        #INFO: セルフアテンションの出力を取得
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        #INFO: self_attention_outputs[0]：セルフアテンションの出力
        # -> [1]はAtteniton weight
        attention_output = self_attention_outputs[0]

        #INFO: デコーダの場合、self_attention_outputs[1:-1]によって隠れ層以外の出力を取得
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            #INFO: 更新されたキーとバリューを取得
            # -> 次回のデコード処理で使用
            present_key_value = self_attention_outputs[-1]
        else:
            #INFO: デコーダでない場合、キャッシュされたキーとバリューは存在しない
            # -> 全て取得
            outputs = self_attention_outputs[1:] 

        #INFO: クロスアテンション初期化
        cross_attn_present_key_value = None

        #INFO: クロスアテンションの処理
        #INFO: Decoder かつ Encoder の隠れ状態が存在する場合
        if self.is_decoder and encoder_hidden_states is not None:
            #INFO: crossattention が定義されていない場合，エラー
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            #INFO: past_key_value[2:] は過去のキーとバリューの状態を含むタプル -> [2]:key, [3]:value
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            #INFO: クロスアテンションの出力を取得
            #INFO: 出力値: 
            # -> cross_attention_outputs[0]  : 更新されたattention_output
            # -> cross_attention_outputs[1]  : attention_weights (if output_attentions=True)
            # -> cross_attention_outputs[-1] : 更新されたクロスアテンションのpresent_key_value
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            #INFO: 更新されたcross_attention_outputをattention_outputに追加
            attention_output = cross_attention_outputs[0]
            #INFO: attention_weights を出力に追加
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            #INFO: クロスアテンションのキャッシュを present_key_value に追加
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        #INFO: FeedForward ネットワークの処理
        #INFO: apply_chunking_to_forward() は大きなテンソルを小さなテンソルに分割して順次処理する
        #INFO: feed_forward_chunk は FeedForward ネットワークを適用する関数
        #INFO: chunk_size_feed_forward は FeedForward ネットワークのチャンクサイズ
        #INFO: seq_len_dim はシーケンス長の次元(基本的に1)
        #INFO: attention_output はセルフアテンションの出力 -> FeedForward ネットワークの入力テンソル
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        #INFO: layer_output を outputs の先頭に追加
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        #INFO: デコーダの場合，present_key_value を outputs に追加
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    #INFO: FeedForward ネットワークの処理
    def feed_forward_chunk(self, attention_output):
        #INFO: 中間層の適用
        intermediate_output = self.intermediate(attention_output)
        #INFO: 出力層の適用 -> 残差接続とLayerNormalization
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output




class RobertaEncoder(nn.Module):
    """
    複数の RobertaLayer を積み重ねる部分
    """

    #INFO: 初回実行
    def __init__(self, config):
        super().__init__()
        self.config = config
        #INFO: レイヤーの定義
        #INFO: レイヤーインスタンスを config.num_hidden_layers だけ生成し，リスト化
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states           : torch.Tensor,
        attention_mask          : Optional[torch.FloatTensor]               = None,
        head_mask               : Optional[torch.FloatTensor]               = None,
        encoder_hidden_states   : Optional[torch.FloatTensor]               = None,
        encoder_attention_mask  : Optional[torch.FloatTensor]               = None,
        past_key_values         : Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache               : Optional[bool]                            = None,
        output_attentions       : Optional[bool]                            = False,
        output_hidden_states    : Optional[bool]                            = False,
        return_dict             : Optional[bool]                            = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
    """
    Parameters
    ----------
    hidden_states (torch.Tensor) : 
        モデルへの入力となる隠れ状態のテンソル。各トークンの特徴表現を含みます。
        size : (batch_size, sequence_length, hidden_size)
    
    attention_mask (Optional[torch.FloatTensor]) :
        パディングされたトークンをマスクするためのテンソル。
        値は `1`（マスクしない）または `0`（マスクする）を取ります。
        これにより、モデルがパディング部分を無視して計算を行います。
        size : (batch_size, sequence_length)

    head_mask (Optional[torch.FloatTensor]) : 
        各アテンションヘッドをマスクするためのテンソル。
        値は `1`（マスクしない）または `0`（マスクする）を取ります。
        特定のアテンションヘッドを無効化する際に使用します。
        size : (num_heads,)

    encoder_hidden_states (Optional[torch.FloatTensor]) : 
        エンコーダの最終層からの隠れ状態。
        デコーダがクロスアテンションを行う際に使用します。
        エンコーダ・デコーダモデルにおいて、デコーダがエンコーダの情報を参照するために必要です。
        size : (batch_size, sequence_length, hidden_size)
    
    encoder_attention_mask (Optional[torch.FloatTensor]) : 
        エンコーダ入力のパディングトークンをマスクするためのテンソル。
        値は `1`（マスクしない）または `0`（マスクする）を取ります。
        デコーダのクロスアテンション時にエンコーダのパディング部分を無視するために使用します。
        size : (batch_size, sequence_length)

    past_key_values (Optional[Tuple[Tuple[torch.FloatTensor]]]) :
        各レイヤーの過去のキーとバリューの状態を含むタプル。
        形状は `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)` で、デコーディング時の計算を高速化するために使用されます。
        これにより、過去の情報を再計算せずに次のトークンの予測が可能になります。
        size : (batch_size, num_heads, sequence_length - 1, embed_size_per_head)

    use_cache (Optional[bool]) : 
        `True` に設定すると、`past_key_values`（過去のキーとバリューの状態）が返され、デコーディング時の高速化に役立ちます。

    output_attentions (Optional[bool]) : 
        `True` に設定すると、各レイヤーのアテンションウェイトを出力します。デバッグや可視化の際に、モデルがどのトークンに注目しているかを確認できます。

    output_hidden_states (Optional[bool]) : 
        `True` に設定すると、各レイヤーの隠れ状態を出力します。モデルの内部表現を解析する際に役立ちます。

    return_dict (Optional[bool]) : 
        `True` に設定すると、出力が辞書型（`BaseModelOutputWithPastAndCrossAttentions`）で返されます。`False` の場合、タプルで返されます。

    Returns
    -------
    Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]
    -> タプル形式 or BaseModelOutputWithPastAndCrossAttentions クラスのインスタンス
    BaseModelOutputWithPastAndCrossAttentions :
    - last_hidden_state (torch.FloatTensor of shape `(batch_size, sequence_length, hidden_size)`):
        モデルの最終層からの隠れ状態のシーケンス。各トークンの最終的な表現を含み、次のタスク（例えば分類や生成）の入力として使用されます。

    - past_key_values (tuple):
        各レイヤーの過去のキーとバリューの状態を含むタプル。デコーディング時に計算を高速化するために使用され、次回の入力に再利用できます。

    - hidden_states (tuple(torch.FloatTensor), *オプション*):
        各レイヤーの隠れ状態を含むタプル。`output_hidden_states=True` の場合に返され、モデルの各層の出力を詳細に解析する際に役立ちます。

    - attentions (tuple(torch.FloatTensor), *オプション*):
        各レイヤーのアテンションウェイトを含むタプル。`output_attentions=True` の場合に返され、モデルがどのトークン間の関係に注目しているかを理解するのに役立ちます。

    - cross_attentions (tuple(torch.FloatTensor), *オプション*):
        各レイヤーのクロスアテンションウェイトを含むタプル。
        エンコーダ・デコーダモデルで `output_attentions=True` の場合に返され、デコーダがエンコーダのどの部分に注目しているかを解析できます。
    """

        #INFO: 中間結果の初期化
        # -> all_hidden_states      : 各レイヤーの隠れ状態を格納
        # -> all_self_attentions    : 各レイヤーのself-attentionを格納
        #       -> output_attentions=True の場合，空のタプルを作成 False の場合，None
        # -> all_cross_attentions   : 各レイヤーのcross-attentionを格納
        #       -> output_attentions=True かつ self.config.add_cross_attention=True の場合，空のタプルを作成 False の場合，None

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        #INFO: 勾配チェックポイントを使用する場合の処理
        # -> 勾配チェックポイントを使用する場合，矛盾が発生するため`use_cache=True` は使用できない
        if self.gradient_checkpointing and self.training:
            #INFO: use_cache=True の場合は警告
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        #INFO: 次のデコーダのキャッシュを初期化(use_cache=True の場合)
        next_decoder_cache = () if use_cache else None



        #!-------------------------------------------------------
        #!-------------------------------------------------------
        #! 提案手法の実装部分！
        #!-------------------------------------------------------
        #!-------------------------------------------------------


        #INFO: 各レイヤーの処理
        #INFO: レイヤーの数だけ処理を繰り返す
        #INFO: enumerate() はインデックスと要素を同時に取得する関数
        for i, layer_module in enumerate(self.layer):
            #INFO: 現在のレイヤーの隠れ状態を保存(output_hidden_states=True の場合)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            #INFO: ヘッドマスクと過去のキー・バリューの取得(head_mask, past_key_values が Noneでない場合)
            # -> デコーダの高速化のため
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            #INFO: レイヤーの順伝播
            #INFO: 勾配チェックポイントの適用（使用する場合）
            if self.gradient_checkpointing and self.training:
                #INFO: 勾配チェックポイントを適用
                #INFO: _gradient_checkpointing_func() は勾配チェックポイントを適用しながら順伝播を行う関数
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,      # -> __call__ : インスタンスを関数のように呼び出す -> foward() が呼び出される
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            #INFO: 勾配チェックポイントを使用しない場合 or 推論時
            else:
                #INFO: 通常の順伝播
                # -> layer_module() は RobertaLayer の forward() メソッド
                # -> __call__を呼び出していないためforward()が呼び出されないのでは？ -> layer_module.forward() で呼び出している
                # -> layer_module.__call__() は layer_module() は本質的に同じ
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
                """
                ! なぜ勾配チェックポイント時に __call__ を明示的に渡すのか
                    -> 勾配チェックポイントは、メモリ使用量を削減するために、一部の中間アクティベーションを保存せず、逆伝播時に再計算する技術です。
                    -> torch.utils.checkpoint.checkpoint 関数は、再計算を行うために関数オブジェクトを必要とします。
                    -> layer_module.__call__ を関数として渡すことで、forward メソッドの実行を制御できます。

                nn.Module を継承したクラスのインスタンス（ここでは layer_module）は、関数のように呼び出すことができます。
                例えば、output = module(input) とすると、内部的に module.__call__(input) が実行されます。
                __call__ メソッドは、様々なフック処理やモジュールの前後処理を行った後、forward メソッドを呼び出します。
                """

            #INFO: レイヤーの出力から隠れ状態を更新
            hidden_states = layer_outputs[0]
            #INFO: レイヤーの出力から次のデコーダのキャッシュを更新(use_cache=True の場合)
            #INFO: layer_outputs[1] は past_key_values
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            #INFO: レイヤーの出力からアテンションを更新(output_attentions=True の場合)
            #INFO: layer_outputs[1] は self-attention
            #INFO: layer_outputs[2] は cross-attention
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        #INFO: 最終的な各レイヤーの隠れ状態を返す
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        #INFO:  return_dict が False の場合，タプル形式で返す
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        
        #INFO: return_dict が True の場合，SequenceClassifierOutput クラスのインスタンスを返す
        return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
        )



class RobertaPooler(nn.Module):
    """
    モデルの出力をプールするためのクラス
    -> 隠れ状態から固定長のベクトルを生成し，タスクに適したヘッドに入力する
    """

    #INFO: 初回実行
    def __init__(self, config):
        super().__init__()
        #INFO: 全結合層(Dence)の定義
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #INFO: 活性化関数(tanh)の定義
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理
        Parameters
        ----------
        hidden_states : torch.Tensor
            モデルの隠れ状態のテンソル
            size : (batch_size, sequence_length, hidden_size)
        
        Returns
        -------
        torch.Tensor
            プールされた出力のテンソル
            size : (batch_size, hidden_size)
        """

        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        #INFO: [CLS] token の隠れ状態を取得
        # -> [CLS] は入力シーケンス全体の要約特徴を持つと仮定されているらしい
        first_token_tensor = hidden_states[:, 0]
        #INFO: 全結合層を適用
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        #INFO: プールされた出力を返す
        # -> 用途：上位モデルや分類器への入力
        return pooled_output

    

class RobertaPreTrainedModel(PreTrainedModel):
    """
    モデルの重みを読み込むためのクラス
    
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RobertaEmbeddings", "RobertaSelfAttention"]

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class RobertaModel(RobertaPreTrainedModel):
    """
    Roberta モデルは Encoder（Attentionのみ）としてもDecoder（Attention + Cross-Attention）としても使用可能
    Decoderとして使用する場合は，`config.is_decoder=True` を設定する必要がある
    Seq_to_Seq モデルを構築する場合は，`is_decoder, add_cross_attention = True` を設定して初期化する
    ->  encoder_hidden_states がフォワードパスへの入力として想定される．

    """

    #INFO: 既存のBERTモデルのコードをコピー: from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config) 
        self.encoder = RobertaEncoder(config)       #! 未定義

        self.pooler = RobertaPooler(config) if add_pooling_layer else None #! 未定義

        #INFO: post_init() によってモデルの重みを初期化
        #INFO: 構成要素[embed, encoder, pooler] を定義しておかないとエラー
        self.post_init()

    
    def get_input_embeddings(self):
        """
        単語埋め込み層への参照を返す
        embeddings.weightを書き換えれば重みの更新ができる
        """
        return self.embeddings.word_embeddings 
        

    def set_input_embeddings(self, value):
        """
        単語埋め込み層の重みを設定
        value : 単語埋め込み層の重み
        """
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        モデルのヘッドを剪定する
        heads_to_prune : 剪定するヘッドの辞書
        PreTrainedModel._prune_heads() をオーバーライド -> 詳細はそっち
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=BaseModelOutputWithPoolingAndCrossAttentions,
    #     config_class=_CONFIG_FOR_DOC,
    # )

    #INFO: 既存のBERTモデルのコードをコピー: from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids               : Optional[torch.Tensor]            = None,
        attention_mask          : Optional[torch.Tensor]            = None,
        token_type_ids          : Optional[torch.Tensor]            = None,
        position_ids            : Optional[torch.Tensor]            = None,
        head_mask               : Optional[torch.Tensor]            = None,
        inputs_embeds           : Optional[torch.Tensor]            = None,
        encoder_hidden_states   : Optional[torch.Tensor]            = None,
        encoder_attention_mask  : Optional[torch.Tensor]            = None,
        past_key_values         : Optional[List[torch.FloatTensor]] = None,
        use_cache               : Optional[bool]                    = None,
        output_attentions       : Optional[bool]                    = None,
        output_hidden_states    : Optional[bool]                    = None,
        return_dict             : Optional[bool]                    = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        

        Parameters
        ----------
        input_ids (Optional[torch.Tensor]) : 入力データのトークンID
            size : (batch_size, sequence_length)
        attention_mask (Optional[torch.Tensor]) : アテンションマスク
            size : (batch_size, sequence_length)
        token_type_ids (Optional[torch.Tensor]) : 各トークンがどのセグメント（文）に属しているかを示すための ID（整数値）のテンソル(文1 [SEP] 文2 の区別をつける)
                                                -> セグメント埋め込みを使用する場合に使用
            size : (batch_size, sequence_length)
        position_ids (Optional[torch.Tensor]) : 位置ID
            size : (batch_size, sequence_length)
        head_mask (Optional[torch.Tensor]) : ヘッドマスク
            size : (num_heads,)
        inputs_embeds (Optional[torch.Tensor]) : 入力埋め込み
            size : (batch_size, sequence_length, hidden_size)
        encoder_hidden_states  (`torch.FloatTensor`、形状は `(batch_size, sequence_length, hidden_size)`、*オプション*):
            エンコーダの最終層の出力における隠れ状態のシーケンス。モデルがデコーダとして設定されている場合、クロスアテンションで使用されます。
        encoder_attention_mask (`torch.FloatTensor`、形状は `(batch_size, sequence_length)`、*オプション*):
            エンコーダ入力のパディングトークンインデックスに対するアテンションを避けるためのマスク。モデルがデコーダとして設定されている場合、クロスアテンションで使用されます。マスクの値は `[0, 1]` のいずれかです：

            - **1**：マスクされて **いない** トークンに対応
            - **0**：マスクされて **いる** トークンに対応
        past_key_values (`tuple(tuple(torch.FloatTensor))`、長さが `config.n_layers` のタプルで、各タプルは形状が `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)` の4つのテンソルを含む):
            アテンションブロックの事前計算されたキーとバリューの隠れ状態を含みます。デコーディングを高速化するために使用できます。

            `past_key_values` を使用する場合、ユーザーは全ての `decoder_input_ids`（形状が `(batch_size, sequence_length)`）の代わりに、最後の `decoder_input_ids`（このモデルに過去のキー・バリュー状態が与えられていないもの、形状が `(batch_size, 1)`）のみを入力することができます。

        use_cache (`bool`、*オプション*):
            `True` に設定すると、`past_key_values`（キーとバリューの状態）が返され、デコーディングの高速化に使用できます（`past_key_values` を参照）。
        output_attentions (Optional[bool]) : アテンションの出力
        output_hidden_states (Optional[bool]) : 隠れ状態の出力
        return_dict (Optional[bool]) : 辞書の出力

        Returns
        -------
        Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]
        -> タプル形式 or BaseModelOutputWithPoolingAndCrossAttentions クラスのインスタンス
        BaseModelOutputWithPoolingAndCrossAttentions には，hidden_states, attentions, cross_attentions が含まれる
        - last_hidden_state (torch.FloatTensor of shape `(batch_size, sequence_length, hidden_size)`):
        -> モデルの最終層からの隠れ状態のシーケンス。
        -> 入力シーケンス内の各トークンに対応するベクトル表現が含まれています。

        - pooler_output (torch.FloatTensor of shape `(batch_size, hidden_size)`):
        -> シーケンス全体の要約表現。通常、最初のトークン（`[CLS]` トークン）の隠れ状態に線形変換と活性化関数（`tanh`）を適用して得られます。
        -> 文章分類などのタスクで使用されます。

        - past_key_values (tuple):
        -> 各層の過去のキーとバリューの隠れ状態を含むタプル。
        -> デコーディング時に計算を高速化するために使用されます。
        -> 次のトークンを生成する際に、過去の情報を再利用できます。

        - hidden_states (tuple(torch.FloatTensor), *オプション*):
        -> 各層の隠れ状態を含むタプル。
        -> `output_hidden_states=True` の場合に返されます。
        -> 各テンソルはそれぞれの層の出力を表し、モデルの内部挙動を詳細に解析する際に役立ちます。

        - attentions (tuple(torch.FloatTensor), *オプション*):
        -> 各層の自己アテンションの重みを含むタプル。
        -> `output_attentions=True` の場合に返されます。
        -> 各テンソルは各層でのアテンションマップを示し、どのトークンが他のトークンに注目しているかを理解するのに役立ちます。

        - cross_attentions (tuple(torch.FloatTensor), *オプション*):
        -> 各層のクロスアテンションの重みを含むタプル。
        -> モデルがデコーダーとして動作する場合に使用され、`output_attentions=True` の場合に返されます。
        -> エンコーダーからの情報に対するアテンションを表し、エンコーダーとデコーダー間の相互作用を解析する際に有用です。
        """


        #INFO: パラメータのデフォルト値設定
        #INFO: デフォルト値が設定されていない場合，config が適用される
        #INFO: 初期化パラメータ：output_attentions, output_hidden_states, return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #INFO: デコーダの場合，エンコーダの隠れ状態を受け取る
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        #INFO: input_ids と inputs_embeds のどちらかが指定されている場合，エラーを返す
        #INFO:   input_ids      : トークン化された単語やサブワードの ID（整数値）。これらはモデル内部で埋め込み層を通してベクトル表現に変換される。
        #INFO:   inputs_embeds  : 既に埋め込みベクトルに変換された入力。つまり、ユーザーが独自に埋め込み層を処理して得た ベクトル表現（連続値） を直接モデルに入力します。
        # -> 両方とも本質は一緒なので，重複しないようにしている
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        #INFO: input_shapeからbatch_size, seq_lengthを取得
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        #INFO: past_key_values_length の初期化
        #INFO: past_key_values を用保持することでDecoderの再計算を避ける
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        #INFO: attention_mask の初期化
        #INFO: attention_mask が None なら1で初期化
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        #INFO: token_type_ids の初期化
        #INFO: token_type_ids が None なら0で初期化
        if token_type_ids is None:
            #INFO: RobertaEmbeddings に token_type_ids がある場合，その値を取得
            #INFO: hasattr() はオブジェクトが指定された属性を持っているかどうかを調べる
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        #INFO: attention_mask の拡張
        #INFO: [batch_size, from_seq_length, to_seq_length]のself-attentionマスクを自分で作成できるよ
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)


        #INFO: encoder_hidden_states の初期化
        #INFO: encoder_hidden_states が None なら0で初期化
        #INFO: cross_attention に2D or 3D アテンションマスクが渡されている場合，[batch_size, num_heads, seq_length, seq_length]にブロードキャストする
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            #INFO: encoder_attention_mask が None なら1で初期化
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        #INFO: 必要に応じてヘッドマスクを準備
        #INFO: head_mask が1.0の場合，ヘッドを保持する
        # -> attention_probs の形状は bsz x n_heads x N x N
        # -> input head_mask の形状は [num_heads] or [num_hidden_layers x num_heads]
        # -> head_mask は [num_hidden_layers x batch x num_heads x seq_length x seq_length] に変換される 
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        #INFO: input -> RobertaEmbeddings -> output までの処理を行う
        #INFO: RobertaEmbeddings の forward 処理
        #INFO: input テンソルに対してEmbbeingsを行い，位置情報を付加する
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        #INFO: RobertaEncoder の forward 処理
        #INFO: Embedding -> Encoder
        #INFO: encoder_outputsは，(sequence_output, pooled_output) となる
        # -> sequence_output : 最後の層の隠れ状態   ->  各単語の隠れ状態
        # -> pooled_output   : CLSトークンの隠れ状態
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        #INFO: sequence_output : 最後の層の隠れ状態 -> 各トークンに対するベクトル表現が格納されている
        sequence_output = encoder_outputs[0]
        #INFO: 出力に対してpooling処理を行う
        #INFO: CLSトークンの隠れ状態を取得
        # -> 線形変換(dense層) -> 活性化関数(tanh) -> 出力
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None


        #INFO:  return_dict が False の場合，タプル形式で返す
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        #INFO: return_dict が True の場合，BaseModelOutputWithPoolingAndCrossAttentions クラスのインスタンスを返す
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class RobertaFor15LayersClassification(RobertaPreTrainedModel):
    """
    最初に呼び出されるclass
    元：RobertaForSequenceClassification(RobertaPreTrainedModel):
    機能：
    - RobertaModel のインスタンス化
        -> Input Embedding を行い，Encoder に通すクラス           
    - RobertaClassificationHead のインスタンス化
        -> CLS token ベクトルからMLPを通して，最終的な出力を得るクラス
    - Forward 処理
        -> 読み込んだクラスを用いて，入力データから出力までの処理を行う
    
    機能まとめ：
    モデルを初期化し，forward処理によって，入力データを受け取り，出力を返す
    """

    #INFO: 初回実行
    def __init__(self, config: RobertaConfig) -> None:
        # --RobertaConfig--
        # vocab_size = 50265
        # hidden_size = 768
        # num_hidden_layers = 12
        # num_attention_heads = 12
        # intermediate_size = 3072
        # hidden_act = 'gelu'
        # hidden_dropout_prob = 0.1
        # attention_probs_dropout_prob = 0.1
        # max_position_embeddings = 512
        # type_vocab_size = 2
        # initializer_range = 0.02
        # layer_norm_eps = 1e-12
        # pad_token_id = 1
        # bos_token_id = 0
        # eos_token_id = 2
        # position_embedding_type = 'absolute'
        # is_decoder = False
        # use_cache = True
        # classifier_dropout = None**kwargs
         
        super().__init__(config)
        # デフォルトで設定されていないので，設定していないとAttributeErrorが発生する…？
        self.num_labels = config.num_labels
        self.config = config

        #INFO: RobertaModel のインスタンス化
        #INFO: Embedding -> Encoder
        self.roberta = RobertaModel(config, add_pooling_layer=False) #<- イマココ！(10/02/:19:43)

        #INFO: RobertaClassificationHead のインスタンス化
        #INFO: CLS vec -> MLP -> logit
        self.classifier = RobertaClassificationHead(config)#! 未定義

        # Initialize weights and apply final processing
        #INFO: 追加の初期化処理
        #INFO: モデル内のサブモジュールを定義してから重みの初期化を行う
        # -> まだ定義してないモジュールの重み初期化を行うとエラーになるから
        self.post_init()

    #DEBUG : 多分いらない
    # @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     checkpoint="cardiffnlp/twitter-roberta-base-emotion",
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output="'optimism'",
    #     expected_loss=0.08,
    # )

    #INFO: forward処理
    #INFO: input -> RobertaModel -> RobertaClassificationHead -> output までの一連の処理を行う
    def forward(
        self,
        input_ids               : Optional[torch.LongTensor]    = None,
        attention_mask          : Optional[torch.FloatTensor]   = None,
        token_type_ids          : Optional[torch.LongTensor]    = None,
        position_ids            : Optional[torch.LongTensor]    = None,
        head_mask               : Optional[torch.FloatTensor]   = None,
        inputs_embeds           : Optional[torch.FloatTensor]   = None,
        labels                  : Optional[torch.LongTensor]    = None,
        output_attentions       : Optional[bool]                = None,
        output_hidden_states    : Optional[bool]                = None,
        return_dict             : Optional[bool]                = None,
    )   -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
    """
    Parameters
    ----------
    labels (`torch.LongTensor`、形状は `(batch_size,)`、*オプション*):
    ->  シーケンス分類/回帰の損失を計算するためのラベル。
        インデックスは `[0, ..., config.num_labels - 1]` の範囲である必要があります。
        `config.num_labels == 1` の場合、回帰損失（平均二乗誤差）が計算されます。
        `config.num_labels > 1` の場合、分類損失（クロスエントロピー）が計算されます。

    Returns
    -------
    Union返り値について
    typing.Union
    ユニオン型： Union[X, Y] は X または Y を表す．
    -> Tuple[torch.Tensor], SequenceClassifierOutput のどちらかを返す

    Tuple[torch.Tensor]     ：`return_dict=False` の場合、モデルの出力はタプル形式で返されます。
                              このタプルには、モデルからの各種出力（例えば、`logits`、`hidden_states`、`attentions` など）が順番に格納されています。
                              インデックスを使用して各出力にアクセスします。
                              例えば、`outputs[0]` は `logits` になります。

    SequenceClassifierOutput：`return_dict=True` の場合、モデルの出力は `SequenceClassifierOutput` というデータクラスのインスタンスとして返されます。
                              このデータクラスは、名前付き属性で各出力にアクセスでき、コードの可読性と保守性が向上します。
                              主な属性には `loss`、`logits`、`hidden_states`、`attentions` などがあります。
                              例えば、`outputs.logits` でロジットにアクセスできます。
    """
    
        #INFO: return_dict が None の場合，config.use_return_dict が適用される
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #INFO: RobertaModel の forward 処理
        #INFO: input -> RobertaModel(Encoder) -> output までの処理を行う
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        #INFO: RobertaClassificationHead の forward 処理
        logits = self.classifier(sequence_output)


        loss = None

        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            #INFO: 損失関数の選択
            #INFO: 損失関数の選択 -> シーケンス分類/回帰の損失を計算するためのラベル
            #INFO: labelが1つの場合は回帰，それ以外は分類
            # 回帰：平均二乗誤差
            # 分類：cross entropy loss
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                # MSE Loss
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # Cross Entropy Loss
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # BCE Loss
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        #INFO:  return_dict が False の場合，タプル形式で返す
        # outputs[0] : logits
        # outputs[1:] : hidden_states, attentions
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        #INFO: return_dict が True の場合，SequenceClassifierOutput クラスのインスタンスを返す
        # outputs.loss
        # outputs.logits
        # outputs.hidden_states
        # outputs.attentions
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaClassificationHead(nn.Module):
    """
    文章分類タスク用のヘッド
    入力：CLSトークンのベクトル表現
    出力：logit
    """

    #INFO: 初回実行
    def __init__(self, config):
        super().__init__()
        #INFO: dense層（中間層）の定義 -> 全結合層（線形変換）
        # -> output = input * weight.T + bias
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #INFO: classification用のdropout率の設定
        # -> 過学習の防止
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        #INFO: dropout層の定義
        self.dropout = nn.Dropout(classifier_dropout)
        #INFO: 出力層の定義
        # -> 最終的な出力を得るための全結合層
        # -> output = input * weight.T + bias
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        """
        2層の全結合層からなるMLP(dence, out_proj)
        MLP層を通して，最終的なlogitを得る
        処理：
        hidden_size -> dense -> tanh -> dropout -> out_proj -> logit
        """
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
















class Embedding(nn.Module):
    """
    単語埋め込みの作成を行う
    入力シーケンス内の各単語を埋め込みベクトルに変換する
    埋め込みベクトルは設定されたモデルの次元数と一致する

    例）
    埋め込みベクトル ：512
    語彙サイズ      ：100
    -> 100 * 512 の埋め込みサイズになる

    出力のサイズはバッチサイズ * シーケンス長 * 埋め込みサイズ

    例）
    バッチサイズ  ：32
    シーケンス長  ：10
    埋め込みサイズ：512
    -> 32 * 10 * 512 のテンソルが出力される

    ! 語彙サイズとシーケンス長の違いは，語彙サイズは辞書のサイズであり，シーケンス長は入力の長さである（それぞれ独立）


    Parameters
    ----------
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """
        初期化

        Parameters
        ----------
            vocab_size (int): 語彙サイズ
            embed_dim (int) : 埋め込み次元サイズ
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        埋め込みベクトルを計算する

        Parameters
        ----------
        x (torch.Tensor): トークン化された単語IDのテンソル
            size        : (batch_size, seq_len)

        Returns
        -------
        torch.Tensor    : 埋め込みベクトル
            各単語IDに対応する埋め込みベクトルを含むテンソル
            size        : (batch_size, seq_len, embed_dim)

        Notes
        -----
        埋め込み層 (embed) は，訓練時に学習されるパラメータに基づいて計算される
        ! 必要に応じて，事前学習されたモデルで初期化することも可能
        """
        return self.embed(x)


# register buffer in Pytorch ->
# If you have parameters in your model, which should be saved and restored in the state_dict,
# but not trained by the optimizer, you should register them as buffers.
# INFO: オプティマイザによって訓練しないパラメータを保存，復元したい場合は，バッファとして登録する必要がある
# 例）self.register_buffer('running_mean', torch.zeros(10)) -> running_mean は訓練されないが，保存される


class PositionalEmbedding(nn.Module):
    """
    位置エンコーディングを行う
    位置エンコーディングは，入力シーケンス内の単語の位置を示すための情報を追加する
    位置エンコーディングは，埋め込みベクトルに加算される形式で保持される

    pos : 文章における単語の順序
    i   : 埋め込みベクトルの順序
    をそれぞれ保持する．


    """

    def __init__(self, max_seq_len: int, embed_model_dim: int) -> None:
        """
        初期化

        Parameters
        ----------
        max_seq_len (int)       : 最大シーケンス長
        embed_model_dim (int)   : 埋め込みモデルの次元数
        """

        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)

        # INFO: 位置エンコーディングの計算
        # INFO: posループによって全シークエンスにおける位置関係を計算
        for pos in range(max_seq_len):
            # INFO: iループによってシーケンス（次元）の順序に対して位置関係を計算
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / self.embed_dim))
                )

        # 例） pe = [[00, 01, 02, ... , 511],
        #          [10, 11, 12, ... , 511],
        #          ...
        #          [99, 98, 97, ... , 512]]
        # ! 理解のために0ではなく行列番号を初期値にしている（実際は全て0埋め）
        # pos : 0, 1, 2, ... , 99  を列ごとに加算していく
        # i   : 0, 1, 2, ... , 511 を行ごとに加算していく
        # 結果：
        # pe = [[00 + 0 + 0, 01 + 0 + 1, 02 + 0 + 2, ... , 511 + 0 + 511],      (pos = 0)
        #       [10 + 1 + 0, 11 + 1 + 1, 12 + 1 + 2, ... , 511 + 1 + 511],      (pos = 1)
        #       ...
        #       [99 + 99 + 0, 98 + 99 + 1, 97 + 99 + 2, ... , 512 + 99 + 511]]  (pos = 99)
        # !　実際のpos, i の値は，sin, cos を用いて計算された値である

        # INFO: テンソルに新しい次元を追加（軸方向）-> 列方向に次元を追加 -> 次元を追加することで，バッチサイズ情報を保存することができる．
        pe = pe.unsqueeze(0)
        # 例） pe = [[[0, 00, 01, 02, ... , 511],
        #            [0, 10, 11, 12, ... , 511],
        #            ...
        #            [0, 99, 98, 97, ... , 512]]]
        # シーケンスの最初の部分にバッチサイズ情報を考慮するための次元を追加している

        # INFO: pe をバッファとして登録 -> パラメータとして保存され，訓練されないが，モデルに保存させることができる
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        位置エンコーディングを計算する

        Parameters
        ----------
        x (torch.Tensor): 埋め込みベクトル (embedding 済み)
            size        : (batch_size, seq_len, embed_dim)

        Returns
        -------
        torch.Tensor    : 位置エンコーディングを追加した埋め込みベクトル
            size        : (batch_size, seq_len, embed_dim)

        Notes
        -----
        埋め込みベクトルに位置エンコーディングを加算する
        """
        # INFO: 埋め込みベクトルに対して埋め込み次元数の平方根を掛ける -> 位置エンコーディングのスケーリング
        # ex  : 512次元 -> 22.62が掛けられる
        x = x * math.sqrt(self.embed_dim)
        # INFO: xのシークエンス長を取得 -> 横方向の長さ（文章あたりの許容量）
        seq_len = x.size(1)
        # INFO: Variable を用いて微分しないことを明示的に示している
        #! Variable は非推奨 -> detach を使用する
        # x = x + torch.autograd.Variable(
        #     self.pe[:, :seq_len], requires_grad=False
        # )
        # INFO: 勾配を計算しないために detach() を使用
        x = x + self.pe[:, :seq_len].detach()

        return x


# ----------------------------------------------------------------------------
# ! Attention
# ----------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """
    Atteniotn の実装


    """

    def __init__(self, embed_dim: int = 512, n_heads: int = 8) -> None:
        """
        初期化

        Parameters
        ----------
        embed_dim (int)     : 埋め込み次元数
        n_heads (int)     : ヘッド数
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        # INFO: ヘッドごとの次元数 -> 512 / 8 = 64, それぞれのquery, key, value の次元数は64となる
        self.single_head_dim = int(
            self.embed_dim / self.n_heads
        )  # embed_dim // n_heads

        # INFO: 線形変換を行う
        # INFO: query, key, value の重み行列を作成
        # INFO: bias は False としている -> バイアス項を追加しない
        self.query_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim, bias=False
        )
        self.key_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim, bias=False
        )
        self.value_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim, bias=False
        )

        # INFO: 出力の重み行列を作成 -> これは512次元
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

    def forward(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        query, key, value に対して 線形変換を行い，Attention を計算する

        Parameters
        ----------
        query : torch.Tensor
            クエリのテンソル (batch_size, seq_len, embed_dim)
        key : torch.Tensor
            キーのテンソル (batch_size, seq_len, embed_dim)
        value : torch.Tensor
            バリューのテンソル (batch_size, seq_len, embed_dim)
        mask : Optional[torch.Tensor], default=None
            マスクテンソル (batch_size, seq_len), optional

        Returns
        -------
        torch.Tensor
            Attentionを計算した後のテンソル (batch_size, seq_len, embed_dim)
        """
        # print("position: MultiHeadAttention -> forward")
        batch_size = key.size(0)
        seq_length = key.size(1)

        # ! decoderの推論時には，queryのシーケンス長が変わる可能性がある
        seq_length_query = query.size(1)

        # INFO: query, key, value の重み行列を計算
        # INFO: view は reshape と同じ -> (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, n_heads, single_head_dim)
        # INFO: transpose は転置 -> (batch_size, seq_len, n_heads, single_head_dim) -> (batch_size, n_heads, seq_len, single_head_dim)
        # INFO: contiguous はメモリ上に連続した領域を確保する
        # INFO: contiguous は view などの操作を行う際に必要
        # INFO: 32 * 10 * 512 -> 32 * 10 * 8 * 64
        # print(
        #     f"Position MultiHeadAttention -> forward -> view : query:{query.shape}, key:{key.shape}, value:{value.shape}"
        # )
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.view(
            batch_size, seq_length_query, self.n_heads, self.single_head_dim
        )
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        # print(f"Debug: check shapes: {key.shape}, {query.shape}, {value.shape}")

        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        # INFO: (batch_size, n_heads, seq_len, single_head_dim) -> 32 * 8 * 10 * 64 -> 各ヘッドに対して計算できるような形状に変換している
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # INFO: Attention を計算
        # INFO: 内積計算のために，key の次元を転置 -> (batch_size, n_heads, seq_len, single_head_dim) -> (batch_size, n_heads, single_head_dim, seq_len)
        # INFO: (32 * 8 * 10 * 64) * (32 * 8 * 64 * 10) -> (32 * 8 * 10 * 10)
        # INFO: transpose(-1, -2) は最後の2つの次元を転置する
        k_adjusted = k.transpose(-1, -2)
        product = torch.matmul(q, k_adjusted)

        # INFO: マスキングを適用
        # INFO: mask が None でない場合，マスクを適用
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        # INFO: key の次元数の平方根でスケーリング
        product = product / math.sqrt(self.single_head_dim)  # / sqrt(64)

        # INFO: Softmax 関数を適用
        scores = F.softmax(product, dim=-1)

        # INFO: value に対して重みを付与
        # INFO: (32 * 8 * 10 * 10) * (32 * 8 * 10 * 64) -> (32 * 8 * 10 * 64)
        scores = torch.matmul(scores, v)

        # INFO: concat の連結
        # INFO: マルチヘッドの結果を1テンソルに結合
        # INFO: (32 * 8 * 10 * 64) -> (32 * 10 * 8 * 64) -> (32 * 10 * 512)
        concat = (
            scores.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length_query, self.single_head_dim * self.n_heads)
        )

        # INFO: 出力行列の線形変換（学習パラメータ）
        # INFO: 各ヘッドの方向性を考慮して，出力を計算
        output = self.out(concat)
        # print(f"Position: MultiHeadAttention -> forward -> output: {output.shape}")
        return output


# ----------------------------------------------------------------------------
# ! Transformer Encoder Architecture
# ----------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Encoder Layer

    処理の流れ：
    1. Attention
    2. 残差結合
    3. 正規化
    4. Feed Forward
    5. 残差結合
    6. 正規化
    7. 出力
    """

    def __init__(
        self, embed_dim: int, expansion_factor: int = 4, n_heads: int = 8
    ) -> None:
        super(TransformerBlock, self).__init__()
        """
        初期化
        """

        # INFO: MultiHeadAttention のインスタンス化
        self.attention = MultiHeadAttention(embed_dim, n_heads)

        # INFO: Layer Normalization のインスタンス化
        #  入力の平均と分散を計算し，正規化を行う
        #  正規化：平均を0，分散を1にする処理
        #  2種類あるのは，Attention と Feed Forward それぞれに対して正規化を行うため
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # INFO: Feed Forward のインスタンス化
        #  全結合層を用いて，非線形変換を行う
        #  2層のMLPを用いて，非線形変換
        #  知識情報の抽出を行う部分
        #  nn.Linear は全結合層, nn.ReLU は活性化関数
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
        )

        # INFO: ドロップアウトのインスタンス化
        # 過学習を防ぐために使用
        # ドロップアウトは，20%の確率でランダムにノードを無効化する
        # 訓練時のみに適用
        # 2種類あるのは，Attention と Feed Forward それぞれに対してドロップアウトを行うため
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(
        self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        forward 処理

        Parameters
        ----------
            key (torch.Tensor)     : キー
            query (torch.Tensor)   : クエリ
            value (torch.Tensor)   : バリュー

        Returns
        -------
            norm2_out (torch.Tensor) : Encoder 処理を全て終えた後のテンソル
        """
        # print("position: TransformerBlock -> forward")
        # print(
        #     f"check shapes: key:{key.shape}, query:{query.shape}, value:{value.shape}"
        # )
        # INFO: Attentionの計算
        # (32 * 10 * 512)
        attention_out = self.attention(key, query, value)
        # INFO: Attention の残差結合
        # Attention の出力と元の入力を足し合わせる
        attention_residual_out = attention_out + value
        # INFO: Attention の正規化
        norm1_out = self.dropout1(self.norm1(attention_residual_out))

        # INFO: Feed Forward
        feed_fwd_out = self.feed_forward(norm1_out)
        # INFO: Feed Forward の残差結合
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        # INFO: Feed Forward の正規化
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out


class TransformerEncoder(nn.Module):
    """
    Encoder 本体

    Parameters
    ----------
        seq_len (int)            : シーケンス長
        embed_dim (int)          : 埋め込み次元数
        num_layers (int)         : レイヤー数
        expansion_factor (int)   : 拡張係数 -> MLPの層をどれだけ拡張するか
        n_heads (int)            : ヘッド数

    Returns
    -------
        Encoder を出力
    """

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        embed_dim: int,
        num_layers: int = 2,
        expansion_factor: int = 4,
        n_heads: int = 8,
    ) -> None:
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        # INFO: TransformerBlock のインスタンス化
        #  レイヤー数分の TransformerBlock を作成
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, expansion_factor, n_heads)
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        # INFO: 文章群の埋め込み
        embed_out = self.embedding_layer(x)
        # INFO: 位置エンコーディング
        out = self.positional_encoder(embed_out)

        # INFO: レイヤーごとに処理
        # INFO: (out, out, out) は key, query, value に対して同じ値を入力している
        for layer in self.layers:
            out = layer(out, out, out)

        # 32 * 10 * 512
        return out


# ----------------------------------------------------------------------------
# ? Transformer Decoder Architecture
# ----------------------------------------------------------------------------


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        # print("position: DecoderBlock -> __init__")
        super(DecoderBlock, self).__init__()

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads

        """
        self.attention = MultiHeadAttention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)

    def forward(self, key, query, x, mask):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           mask: mask to be given for multi head attention
        Returns:
           out: output of transformer block

        """
        # print("position: DecoderBlock -> forward")
        # we need to pass mask mask only to fst attention
        attention = self.attention(x, x, x, mask=mask)  # 32x10x512
        value = self.dropout(self.norm(attention + x))
        # print("Position: DecoderBlock -> forward -> tf block")
        out = self.transformer_block(key, query, value)

        return out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        embed_dim,
        seq_len,
        num_layers=2,
        expansion_factor=4,
        n_heads=8,
    ):
        # print("position: TransformerDecoder -> __init__")
        super(TransformerDecoder, self).__init__()
        """
        Args:
           target_vocab_size: vocabulary size of taget
           embed_dim: dimension of embedding
           seq_len : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention

        """
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=4, n_heads=8)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):
        """
        Args:
            x: input vector from target
            enc_out : output from encoder layer
            trg_mask: mask for decoder self attention
        Returns:
            out: output vector
        """
        # print("position: TransformerDecoder -> forward")
        # print(
        #     f"Debug: check shapes: x:{x.shape}, enc_out:{enc_out.shape}, mask:{mask.shape}"
        # )

        x = self.word_embedding(x)  # 32x10x512
        x = self.position_embedding(x)  # 32x10x512
        x = self.dropout(x)

        for layer in self.layers:
            # print(f"Position: TransformerDecoder -> forward -> layer")
            x = layer(enc_out, x, enc_out, mask)
            # print("------------------------------")

        out = F.softmax(self.fc_out(x))
        # print(f"Finihsed Decoder: {out.shape}")
        return out


class Transformer(nn.Module):
    """
    Transformer model全体の流れを定義する
    """

    def __init__(
        self,
        embed_dim,
        src_vocab_size,
        target_vocab_size,
        seq_length,
        num_layers=2,
        expansion_factor=4,
        n_heads=8,
    ):
        super(Transformer, self).__init__()

        """


        """

        self.target_vocab_size = target_vocab_size

        self.encoder = TransformerEncoder(
            seq_length,
            src_vocab_size,
            embed_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            n_heads=n_heads,
        )
        self.decoder = TransformerDecoder(
            target_vocab_size,
            embed_dim,
            seq_length,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            n_heads=n_heads,
        )

    def make_trg_mask(self, trg):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask

    def decode(self, src, trg):
        """
        for inference
        Args:
            src: input to encoder
            trg: input to decoder
        out:
            out_labels : returns final prediction of sequence
        """
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        # print("Position: Transformer -> decode -> enc_out:", enc_out.shape)
        out_labels = []
        batch_size, seq_len = src.shape[0], src.shape[1]
        # outputs = torch.zeros(seq_len, batch_size, self.target_vocab_size)
        out = trg
        for i in range(seq_len):  # 10
            # print(
            #     f"Debug: check all shapes: out:{out.shape}, enc_out:{enc_out.shape}, trg_mask:{trg_mask.shape}"
            # )

            out = self.decoder(out, enc_out, trg_mask)  # bs x seq_len x vocab_dim
            # taking the last token
            out = out[:, -1, :]

            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out, axis=0)

        return out_labels

    def forward(self, src, trg):
        """
        Args:
            src: input to encoder
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)

        outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs

# 最初に呼び出されるやつ
class Roberta15LayersClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_info=None, # ! add
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attn_info=attn_info, # ! add
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

