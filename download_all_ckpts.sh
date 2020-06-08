# download all TF and PyToch checkpoints
declare -a ckpts_set=("B4-4-4H768-ELEC" "B6-3x2-3x2H768-ELEC" "B6-6-6H768-ELEC" "B8-8-8H1024-ELEC" "B10-10-10H1024-ELEC")
declare -a pack_set=("PT" "TF")
mkdir -p ckpts
cd ckpts
for ckpt in "${ckpts_set[@]}"
do
  for pack in "${pack_set[@]}"
  do
    wget -c http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/${ckpt}-${pack}.tar.gz
    tar -xvf ${ckpt}-${pack}.tar.gz
  done
done
cd ..