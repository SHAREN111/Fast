bash base.sh raw 0
bash base.sh raw 1
bash base.sh raw 2
bash base.sh raw 3
bash base.sh raw 4



bash base.sh fastvar_50_60_100_100 0 fastvar
bash base.sh fastvar_50_60_100_100 1 fastvar
bash base.sh fastvar_50_60_100_100 2 fastvar
bash base.sh fastvar_50_60_100_100 3 fastvar
bash base.sh fastvar_50_60_100_100 4 fastvar



bash base.sh sparsevar_0.75 0 sparsevar 0.75
bash base.sh sparsevar_0.75 1 sparsevar 0.75
bash base.sh sparsevar_0.75 2 sparsevar 0.75
bash base.sh sparsevar_0.75 3 sparsevar 0.75
bash base.sh sparsevar_0.75 4 sparsevar 0.75

bash base.sh sparsevar_0.8 0 sparsevar 0.8
bash base.sh sparsevar_0.8 1 sparsevar 0.8
bash base.sh sparsevar_0.8 2 sparsevar 0.8
bash base.sh sparsevar_0.8 3 sparsevar 0.8
bash base.sh sparsevar_0.8 4 sparsevar 0.8

bash base.sh sparsevar_0.85 0 sparsevar 0.85
bash base.sh sparsevar_0.85 1 sparsevar 0.85
bash base.sh sparsevar_0.85 2 sparsevar 0.85
bash base.sh sparsevar_0.85 3 sparsevar 0.85
bash base.sh sparsevar_0.85 4 sparsevar 0.85

bash base.sh sparsevar_0.9 0 sparsevar 0.9
bash base.sh sparsevar_0.9 1 sparsevar 0.9
bash base.sh sparsevar_0.9 2 sparsevar 0.9
bash base.sh sparsevar_0.9 3 sparsevar 0.9
bash base.sh sparsevar_0.9 4 sparsevar 0.9

bash base.sh sparsevar_0.95 0 sparsevar 0.95
bash base.sh sparsevar_0.95 1 sparsevar 0.95
bash base.sh sparsevar_0.95 2 sparsevar 0.95
bash base.sh sparsevar_0.95 3 sparsevar 0.95
bash base.sh sparsevar_0.95 4 sparsevar 0.95




wait


python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/dongchengqi/gpu.py


