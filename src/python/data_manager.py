import pandas as pd
from pathlib import Path
from logging import getLogger, StreamHandler, Formatter
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import re
from typing import Dict, List

STOPWORDS = ["特にな" "特に無"]

LABEL_MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}

class DataProcessing:
    def __init__(self, args:Dict[str, object]):
        self.logger = args["logger"].getChild(__name__)
        self.logger.info(f"setup Child logger : {__name__}")
        self.root_path = args["DATA_PATH"]
        self.key = args["key"]
        self.text = args["text"]
        self.label = args["label"]
        self.mode = args["mode"]
        self.split_rate = args["split_rate"]
        self.seed = args["seed"]
        #INFO ディレクトリ内にReflectionフォルダがあるか確認、あるならパスを設定
        self.reflection_path = Path(self.root_path) / "Reflection"
        if not self.reflection_path.is_dir():
            self.logger.warning(f"{self.reflection_path} does not exist.")
        #INFO Gradeフォルダがあるか確認、あるならパスを設定
        self.grade_path = Path(self.root_path) / "Grade"
        if not self.grade_path.is_dir():
            self.logger.warning(f"{self.grade_path} does not exist.")
        

    def __call__(self) -> pd.DataFrame:
        left_df = self.read_folder(self.reflection_path)
        right_df = self.read_folder(self.grade_path)
        
        df = pd.merge(left_df, right_df, on=self.key, how="inner")

        #INFO modeに応じて分割形式を変更して返す
        return  self.sprit_data_for_user(df, 
                                        key=self.key, 
                                        text=self.text, 
                                        label=self.label, 
                                        mode = self.mode,
                                        split_rate=self.split_rate, 
                                        seed=self.seed)

    def read_folder(self, path: Path, rules: str = "*.csv") -> pd.DataFrame:
        """
        フォルダ内のファイルを全て読み込み，縦方向に結合する

        Parameters
        ----------
        path  : str
            フォルダパス
        rules : str
            読み込むフォルダの形式（デフォルトではcsvファイルすべて）

        Returns
        -------
        df : DataFrame
            読み込んだ全てのcsvを縦に結合したdf.
        """
        self.logger.info(f"Read {path} {rules} data...")
        file_list = list(path.glob(rules))
        self.logger.info(f"Found files: {file_list}")

        df = pd.DataFrame()
        if file_list:
            for file in file_list:
                temp_df = pd.read_csv(file)
                df = pd.concat([df, temp_df], axis=0, ignore_index=True)
            self.logger.info(f"Total rows: {len(df)}")
        else:
            self.logger.error(f"No {rules} files in {path}")

        return df

        #TODO 動作確認後消す
        # data_path = Path(path)
        # self.logger.info(f"read {data_path} {rules} data...")

        # file_list = list(data_path.glob(rules))
        # self.logger.info(f"read : {file_list}")

        # df = pd.DataFrame()
        # if len(file_list) > 0:
        #     for i in file_list:
        #         temp_df = pd.read_csv(i)
        #         df = pd.concat([df, temp_df], axis=0, ignore_index=True)
        #     self.logger.info(f"get {len(df)} rows.")
        # else:
        #     self.logger.error(f"No {rules} files in {data_path}")

        # return df


    def data_clean(self,
                    df:pd.DataFrame,
                    stopwords:List[str]=STOPWORDS, 
                    text:str="answer_content")->pd.DataFrame:
        """
        テキストデータのクリーニングを行う
        （stopwordsの除去, 頻繁過ぎる文章の削除, 長すぎる / 短すぎる文章の削除）

        Parameters
        ----------
        df          : DataFrame
            対象df
        stopwords  : list(str)
            除去する単語のリスト

        Returns
        -------
        cleaned_df : DataFrame
            前処理後のdf
        """
        
        #INFO: NaNを削除
        cleaned_df = df.dropna(subset=[text])

        #INFO: 長すぎる行，短すぎる行を削除
        # cleaned_df = cleaned_df.loc[cleaned_df[text].str.len() > 8]
        # cleaned_df = cleaned_df.loc[cleaned_df[text].str.len() < 1000]

        # stopwordsを含み，短すぎる行を削除
        cleaned_df = cleaned_df.loc[
            ~(
                cleaned_df[text].str.contains("|".join(stopwords))
                & (cleaned_df[text].str.len() <= 16)
            )
        ]

        self.logger.info(f"df shape : {df.shape}")
        self.logger.info(f"cleaned_df shape : {cleaned_df.shape}")
        self.logger.info(f"dumped {df.shape[0] - cleaned_df.shape[0]} rows.")

        #TODO 有効性確認
        # df_reset = df.reset_index(drop=True)
        # cleaned_df = cleaned_df.reset_index(drop=True)

        #INFO: 削除された行の可視化
        mask = ~df[text].isin(cleaned_df[text])
        dump_df = df[mask]
        self.logger.info(f"dump_df shape : {dump_df.shape}")

        #INFO: stopwordsの除去
        cleaned_df[text] = cleaned_df[text].str.replace(
            r"[、。！？｡･ﾟ＇＂「」『』（）《》【】〔〕…―ー－／＼〜〝〟〈〉]", "", regex=True
        )

        return cleaned_df, dump_df



    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        #INFO: 'userid' と 'grade' を固定して残りの列を縦に並べる
        df_melted = df.melt(
            id_vars=["userid", "grade"], var_name="text_label", value_name="text"
        )

        #INFO: 'text_label' から 'course_number' と 'question_number' を抽出
        df_melted[["course_number", "question_number"]] = df_melted["text_label"].str.split(
            "-", expand=True
        )

        #INFO: 'text_label' 列を削除
        df_melted = df_melted.drop("text_label", axis=1)

        #INFO: 欠損値を削除
        df_melted = df_melted.dropna()

        #INFO: df['label'] を補完
        df_melted["label"] = df_melted["grade"].map(LABEL_MAPPING).astype(int)

        return df_melted


    def sprit_data_for_user(self, 
                            df:pd.DataFrame,
                            key:str,
                            text:str, 
                            label:str, 
                            mode:str="all",
                            split_rate : List[float] =[0.8, 0.2, 0], 
                            seed:int=42) -> pd.DataFrame:
        """
        データフレームを特定の列を基準に分割する．

        Parameters
        ----------
        df          : DataFrame
            対象df
        key         : str
            対象列名
        split_rate  : list(float)
            train, valid, testの割合
        seed        : int
            seed値

        Returns
        -------
        train_df : DataFrame
        valid_df : DataFrame
        test_df  : DataFrame
            testはuse_Test = True としたときのみ出力
        """

        #INFO 講義番号，質問番号ごとに列を作成
        def create_column_name(row):
            return f"{int(row['course_number']):02d}-{row['question_number']}"

        #INFO: ノイズ除去
        df, _ = self.data_clean(df, text=text)

        #INFO: 新しい列を作成
        df["new_column"] = df.apply(create_column_name, axis=1)

        #INFO: 新しいデータフレームを定義
        df_pivoted = df.pivot_table(
            index=[key, label],
            columns="new_column",
            values=text,
            aggfunc=lambda x: " ".join(str(item) for item in x),
            #TODO 動作確認後消す
            # aggfunc=lambda x: " ".join(x),
        ).reset_index()
        self.logger.debug(f"columns: {df_pivoted.columns}")

        #INFO: useridごとに変換した結果を表示
        self.logger.debug(f"Pivot Info : {df_pivoted.shape}")
        self.logger.debug(f"Pivot columns : {df_pivoted.columns}")

        #INFO:modeを基準に分割
        #INFO: mode:all は全てのデータを返す
        if mode == "all":
            return self.decode(df_pivoted) 
        else :
            #INFO trainの割合を基準に分割
            train_df, tmp_df = train_test_split(
                df_pivoted,
                train_size=split_rate[0],
                random_state=seed,
                stratify=df_pivoted[label],
                shuffle=True,
            )

        #INFO: mode:train-valid の場合はtrain, validを返す
        if mode == "train-valid":
            return self.decode(train_df), self.decode(tmp_df)
        #INFO: mode:train-valid-test の場合はtrain, valid, testを返す
        elif mode == "train-valid-test":
            valid_size = split_rate[1] / (split_rate[1] + split_rate[2])
            valid_df, test_df = train_test_split(
                tmp_df,
                test_size=1 - valid_size,
                random_state=seed,
                stratify=tmp_df[label],
                shuffle=True,
            )
            return self.decode(train_df), self.decode(valid_df), self.decode(test_df)


#INFO 生徒ごとに1行にまとめる処理を関数として定義しておく（debug用）
def encode(df, key="userid", label="label", text="text",logger=None, filter=None):
    #filterがある場合、その数値と一致する列"question_number"以外の行を削除
    if filter:
        df = df[df["question_number"] == filter]

    #INFO 講義番号，質問番号ごとに列を作成
    def create_column_name(row):
        return f"{int(row['course_number']):02d}-{row['question_number']}"

    #INFO: ノイズ除去
    #DEBUG: データクリーニングは実行済みとする
    # df, _ = data_clean(df, text=text)

    #INFO: 新しい列を作成
    df.loc[:, "new_column"] = df.apply(create_column_name, axis=1)
    #! 可読性向上のため上に変更 - 動作確認語削除
    # df["new_column"] = df.apply(create_column_name, axis=1)

    #INFO: 新しいデータフレームを定義
    df_pivoted = df.pivot_table(
        index=[key, label],
        columns="new_column",
        values=text,
        aggfunc=lambda x: " ".join(str(item) for item in x),
        #TODO 動作確認後消す
        # aggfunc=lambda x: " ".join(x),
    ).reset_index()
    logger.debug(f"columns: {df_pivoted.columns}")

    #NaNを特殊トークンに置換
    df_pivoted = df_pivoted.fillna("[NA]")

    return df_pivoted
