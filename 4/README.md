# Atividade 4 - Modelagem

Sua tarefa nesta atividade será modificar o controle remoto desenvolvido na primeira atividade para executar movimentos mais precisos que serão utilizados nas atividades seguintes. Ao invés de transformamos comandos do teclado diretamente em valores de tensão elétrica como na [Atividade 1](../../manual/README.md), você primeiro deve agora transformar os comandos de teclado em sinais de controle de velocidade $`v`$ e taxa de rotação $`\omega`$, que então são transformados em sinais de tensão a serem enviados ao robô. Isso é feito utilizando o modelo inverso de dinâmica visto em aula:

```math
\begin{cases}
  V_l=\frac{1}{R}(K_m-K_t)(v-\omega L)\\
  V_r=\frac{1}{R}(K_m+K_t)(v+\omega L).
\end{cases}
```

Na equação acima, $`V_l`$ e $`V_r`$ são as tensões elétricas a serem enviadas para o motor esquerdo e direito, respectivamente, a fim de que o robô se desloque com uma dada velocidade $`v`$ e taxa de rotação $`\omega`$. 
As constantes $`R`$ e $`L`$ indicam o raio da roda e a distância entre o centro do robô e a roda, respectivamente. 
No robô real, tais constanes seriam medidas. Adote os valores $`R=0.0318`$ e $`2L=0.102`$ [m] que são próximos dos valores nominais de construção do Duckiebot. O simulador usa valores próximos a esses, com algum erro aleatório para gerar o efeito de medições imprecisas.
Em um robô ideal, as rodas e motores são idênticos e possuem a mesma constante $`K_m`$ relacionando tensão elétrica e torque. 
No robô real, no entanto, as rodas e motores possuem discrepâncias que fazem com que essa relação seja distinta para cada roda.
A constante $`K_t`$ provê um grau de liberdade adicional para corrigir tais discrepâncias.

## Dinâmica inversa

O primeiro passo é implementar a função `get_pwm_control` no arquivo [agent.py](./agent.py) que recebe valores de velocidade e taxa de rotação e os converte em valores de tensão para as rodas esquerda e direita do robô usando a equação acima. Note que você deve utilizar as constantes deconhecidas $`K_m`$ e $`K_t`$. Por ora, adote os valores arbitrários fornecidos.

## Estimando constantes

Com a função de dinâmica inversa implementada, você deve realizar os seguintes experimentos empíricos a fim de determinar as constantes $`K_m`$ e $`K_t`$.

1. Com um valor arbitrário de $`K_m`$, encontre $`K_t`$ comandando valores de tensão idênticos em ambos os motores ($`V_l=V_r=c`$) e ajustando seu valor para que o robô ande em linha reta. Note que o simulador informa a pose do robô. Você pode usar essa informação para determinar o valor da constante a partir do experimento.
2. Com a constante $`K_t`$ fixa, encontre o valor de $`K_m`$ para que o robô se mova em linha reta por exatamente 1 unidade de distância (o simulador usa uma unidade arbitrária para medir a pose do robô).

Assegure-se que os valores encontrados para as constantes estão corretos realizando o movimento como girar ao redor do seu eixo (ou seja, sem sair do lugar).

## Controle por velocidade e rotação

Com as constantes estimadas, modifique a função `send_commands` do agente para que os comandos do teclado sejam convertidos em comandos de velocidade e ângulo (por exemplo, a tecla "esquerda" deve fazer o robô girar no seu eixo em sentido anti-horário, e a tecla "acima" deve fazer o robô se mover em velocidade constante para a frente). Seu programa deve converter tais ações em comandos a serem enviados ao robô através da função `get_pwm_control`.

## Submissão

Submeta o seu código `agent.py` pelo e-disciplinas.
