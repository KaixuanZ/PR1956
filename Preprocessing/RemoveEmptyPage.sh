# Declare ad pages
declare -a AdPage=("pr1956_f0046_8_0.json" "pr1956_f0081_2_1.json" "pr1956_f0063_2_1.json" "pr1956_f0180_2_0.json"
                    "pr1956_f0063_3_0.json" "pr1956_f0097_3_1.json" "pr1956_f0097_4_0.json" "pr1956_f0128_2_1.json"
                    "pr1956_f0128_3_0.json" "pr1956_f0150_1_1.json" "pr1956_f0150_2_0.json" "pr1956_f0163_2_1.json"
                    "pr1956_f0163_3_0.json" "pr1956_f0174_1_1.json" "pr1956_f0174_2_0.json" "pr1956_f0180_1_1.json")

Path='../../results/personnel-records/1956/seg/firm/page_rect/'

echo "Removing Empty Page"

for json in "${AdPage[@]}"; do
  echo "Removing ${Path}$json"
  rm ${Path}$json
done


declare -a AdPage=("pr1956_f0184_2_0.json")

Path='../../results/personnel-records/1956/seg/supplement/page_rect/'

echo "Removing Empty Page"

for json in "${AdPage[@]}"; do
  echo "Removing ${Path}$json"
  rm ${Path}$json
done