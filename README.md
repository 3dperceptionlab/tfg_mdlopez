# TFG mdlopez

## Docker
- Imagen especificada en el fichero Dockerfile, partiendo de huggingface/transformers-pytorch-gpu:latest e instalando las dependencias necesarias.
- Ejecución usando el scripts de bash `run_docker.sh` que se encarga de montar el volumen con el código fuente y ejecutar el contenedor.

## Ejecución
bash scripts/run-train.sh configs/default.yaml

configurar los parametros de default.yaml:
- name: nombre del experimento (por ejemplo vivit-con-señales)
- eval: True si se quiere evaluar el modelo, False si se quiere entrenar (de momento eval no funciona)
- video_path: si no hay nada o se comenta solo se ejecuta con señales
- signals: path de las señales, comentar para probar unimodal video
- task: clasificacion binaria o multiclase (de momento multiclase)
- video_transformer: los 3 modelos a probar

Resto no hace falta modificar

Login wandb:
Ejecutar -> wandb login
Pegar clave de la web

Comando para abrir terminal:
docker attach tfg_mdlopez# tfg_mdlopez
