# 15-Encoders

## 15回分の入力に対して順番にEncoder-layerを割り当て，総合的な特徴抽出を目指す

###実装済み機能
- config設定
- Encoder のレイヤーをinputに合わせて修正
- hidden_stateの逐次更新
- hidden_stateの逐次保存

###考えていること
- 1回答につき1レイヤーは多分少ない
  - 各レイヤーをもっと増やす！
- 以前の特徴がどんどん薄れていく
  - skip-connectionの実装
