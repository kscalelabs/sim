#!/bin/zsh

echo "If any of the downloads fail, you can manually download the assets and place them in the respective directories."

# Stompymini
echo
echo "Downloading Stompymini assets..."
gdown --id 1Ah92CTN63M2h4uKPLf8eXkX2H9Bg3Kji
unzip meshes.zip -d sim/resources/stompymini/
rm meshes.zip

# Stompypro
echo
echo "Downloading Stompypro assets..."
gdown --folder https://drive.google.com/drive/folders/1-iIqy8j4gF6JeuMc_MjxkRe4vSZl8Ozp -O sim/resources/stompypro/

# Xbot
echo
echo "Downloading Xbot assets..."
gdown --id 1tpl95OdUhg9VL88FhKWMReY4qtLHHKoh
tar -xzvf meshes.zip -C sim/resources/xbot/
rm meshes.zip

# Dora
echo
echo "Downloading Dora assets..."
gdown --folder https://drive.google.com/drive/folders/1tQiMtOwGg3PGo9AygX3sj6X_HBpnW2S_ -O sim/resources/dora/

# G1
echo
echo "Downloading G1 assets..."
gdown --folder https://drive.google.com/drive/folders/1OxYcIJpeih89NY5auRayxCXpndrRnLJx -O sim/resources/g1/

# H1_2
echo
echo "Downloading H1_2 assets..."
gdown --id 19ih7zG6Ky8xJVJD5M1th2hmqtxaNiZyh
tar -xzvf meshes.zip -C sim/resources/h1_2/
rm meshes.zip

# Stompymicro
echo
echo "Downloading Stompymicro assets..."
gdown --folder https://drive.google.com/drive/folders/1C_v0FKoc6um0tUK2f1e6cWXtfvuc-xsD -O sim/resources/stompymicro/

