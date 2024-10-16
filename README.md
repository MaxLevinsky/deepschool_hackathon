# deepschool_hackathon

1. git clone https://github.com/MaxLevinsky/deepschool_hackathon.git
2. cd deepschool_hackathon
3. sudo bash Docker/docker_build.sh
4. sudo Docker/docker_run.sh
тут будут заданы 3 вопроса
    1. вставить путь к директории к dataset.yaml , файл должен называться **dataset.yaml**
"""
            train: ./train/
            val: ./val/
            test: ./test/
            nc: 10
            names: {0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}
"""
       
    3. путь к директории куда сохранять результаты
    4. путь к директории с моделькой *.pt
