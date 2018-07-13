sudo nvidia-docker run -it -p 8888:8888 -p 6006:6006 -v /:/Data  tensorflow/tensorflow:gpu-py3-conda3-rl-imageTools bash

#Apos abrir o docker, ir para pasta /Data. Para usar gpu, o tensorflow gpu esta instalando em um dos env do conda. 
#conda info --envs
#source activate opensim-rl
