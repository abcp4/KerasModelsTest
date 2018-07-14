sudo nvidia-docker run -it -p 8888:8888 -p 6006:6006 -v /:/Data  tensorflow/tensorflow:gpu-py3-conda3-rl-imageTools bash

cd ../Data/media/lsdi/66E03D51E03D2927/keras/

#Apos abrir o docker, ir para pasta /Data. Para usar gpu, o tensorflow gpu esta instalando em um dos env do conda. 
#conda info --envs
#source activate opensim-rl
