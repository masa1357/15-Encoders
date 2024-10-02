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
