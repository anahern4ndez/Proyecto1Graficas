# Proyecto1Graficas


NOTA MUY IMPORTANTE:

Los modelos del perro dálmata, del pato y del cerdito son peculiarmente pesados y se tardan bastante en renderizar (alrededor de 20 min c/u o un poco más, por lo menos en mi computadora), sí renderizan, pero es de tenerles paciencia. Esto se debe a que, en el caso del dálmata, su shader utiliza ruido - lo cual tiene muchas muchas operaciones matemáticas en el trasfondo que ocasionan que se tarde mucho, agregándole el hecho que tiene 9,000 caras; en el caso del cerdito, se debe a que éste tiene 12,000 caras. En mi computadora, el renderizarlo todo tarda más o menos una hora, por lo cual la imagen de todo estará solo en el repositorio GitHub, no en el zip de la entrega (porque quise probar una cosa y se sobreescribió la imagen que tenía de la escena y ya no me da tiempo volverla a renderizar).  Por esto, de preferencia, calificar lo que se encuentra en GitHub. 

Sin embargo, se incluye la imagen de la escena sin estos tres modelos, que son pesados; se incluyen solo los que renderizan relativamente rápido. En el principal.py se renderiza cada modelo con la instrucción de s.bm.load(), si se quiere renderizar modelos individualmente, comentar los demás loads y dejar solo el que se desea. El tiempo de renderizado para cada cosa es el siguiente: 

1. Background (imagen de granja): 1 min.
2. Modelo de la gallina: 40 seg
3. Modelo del gato: 1min 35 seg 
4. Modelo del pato: 10 min (aprox)
5. Modelo del dálmata: > 20 min
6. Modelo del cerdito: > 20 min