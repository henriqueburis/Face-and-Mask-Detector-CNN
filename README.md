# Face-and-Mask-Detector-CNN

By [Luiz H. Buris](http://)

##Introdução

A epidemia de COVID-19 interrompeu rapidamente nossas vidas cotidianas, afetando o comércio e os movimentos internacionais. Usar uma máscara facial para proteger o rosto tornou-se o novo normal. Em um futuro próximo, muitos prestadores de serviços públicos esperam que os clientes usem máscaras adequadamente para participar de seus serviços. Portanto, a detecção de máscara facial tornou-se um dever crítico para ajudar a civilização mundial.

Neste entendimento este trabalho fornece uma maneira simples de alcançar este objetivo utilizando de ferramentas fundamentais de visão computacional omo Aprendizado profundo clássico como redes neurais artificiais, Pytorch e OpenCV.

A técnica sugerida reconhece com sucesso o rosto na imagem ou vídeo e, em seguida, determina se ele tem ou não uma máscara. Como um angente de trabalho de vigilância, ele também pode reconhecer um rosto com uma máscara em movimento, bem como em um vídeo. A técnica atinge excelente precisão. Investigamos valores de parâmetros ótimos para o modelo de Rede Neural Convolucional (CNN) de modo a identificar a existência de máscaras com precisão sem gerar overfitting.

## Code organization

- `CNN`: .


- `facedetcta.py`: .


## Train CNN 
you can now carry out "run" the python scrypt with the following command:

```sh
python3 CNN.py --dataset "face"  --model "vgg19" --train Mask_set/train --test Mask_set/test --n_classe 2 --input_size 32 --epoch 20

```


```sh
python3 facedetcta.py 

```
