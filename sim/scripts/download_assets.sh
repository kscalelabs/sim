#!/bin/zsh

# Stompymini
gdown --id 1Ah92CTN63M2h4uKPLf8eXkX2H9Bg3Kji
unzip meshes.zip -d sim/resources/stompymini/
rm meshes.zip

# Stompypro
gdown --folder https://drive.google.com/drive/folders/1-iIqy8j4gF6JeuMc_MjxkRe4vSZl8Ozp -O sim/resources/stompypro/

# Xbot
gdown --id 1tpl95OdUhg9VL88FhKWMReY4qtLHHKoh
tar -xzvf meshes.zip -C sim/resources/xbot/
rm meshes.zip

# Dora
gdown --folder https://drive.google.com/drive/folders/1tQiMtOwGg3PGo9AygX3sj6X_HBpnW2S_ -O sim/resources/dora/

# G1
gdown --folder https://drive.google.com/drive/folders/1OxYcIJpeih89NY5auRayxCXpndrRnLJx -O sim/resources/g1/

# H1_2
gdown --id 19ih7zG6Ky8xJVJD5M1th2hmqtxaNiZyh
tar -xzvf meshes.zip -C sim/resources/h1_2/
rm meshes.zip

# Stompymicro
gdown --folder https://drive.google.com/drive/folders/1tBiCs3MVzJiMCei72VtVh9GD3x7Zs0XS -O sim/resources/stompymicro/

