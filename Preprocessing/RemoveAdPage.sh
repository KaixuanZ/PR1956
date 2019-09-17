
# Declare ad pages
declare -a AdPage=("pr1956_f0063_2_1.json"
                        "pr1956_f0063_3_0.json"
                        "pr1956_f0081_2_1.json"
                        "pr1956_f0097_3_1.json"
                        "pr1956_f0097_4_0.json"
                        "pr1956_f0128_2_1.json"
                        "pr1956_f0128_3_0.json" )

Path=${Path:-'../../results/personnel-records/1954/seg/page_rect/'}

read -p "Do you want to remove advertisement page in $OutputPath? (y/n) " -n 1 -r
echo -e "\n"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Removing Advertisement Page"

    for json in "${AdPage[@]}"; do
      echo "Removing ${Path}$json"
      rm ${Path}$json
    done
if