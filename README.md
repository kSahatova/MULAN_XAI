# XAI_Med_Vision
Explainable AI for Medical Object Detection

Данное исследование было направлено на изучение методов интерпретации моделей, решающих задачу детекции объектов на медицинских снимках. 
Часть целевого набор данных, который был использован, можно скачать по ссылке [DeepLesion Dataset]:https://www.kaggle.com/datasets/kmader/nih-deeplesion-subset. Изображения представлены в виде КТ снимков для конкретного пациента и были загружены в репозиторий в формате NIFTI. В наборе данных снимков представлены 8 классов, котрые описывают различные части тела. Но основных классов всего два: "нет опухоли", "опухоль". В серии снимков, принадлежащих конкретному пациенту, размечены всего одно или два изображения согласно стандарту DICOM.   
Ввиду специфики распределения набора данных, стандартные предобученные модели такие как FasterRCNN, YOLOv5 и RetinaNet показывают низкую точность детекции. Для экспериментов была использована модель [MULAN]:https://arxiv.org/abs/1908.04373 на основе MaskRCNN, обученная на данном наборе. 


## Окружение
1. Для запуска модели необходима ОС Ubuntu, python>=3.8, CUDA >= 10.2.
2. Все зависимости перечислены в файлах ```requirements.txt``` 

## Использование репозитория
1. Чтобы загрузить модель, необходимо изменить путь до файла аннотации ```DL_info.csv```, котрый находится в директории ```model/data/DeepLesion/DL_info.csv```, в файле ```model/maskrcnn/config/paths_catalog.py```. Также следует изменить пути до подкаталогов в разделе *Misc options* файла ```model/maskrcnn/config/defaults.py```.
2.  Необходимо создать папку ```model/checkpoints``` и загрузить веса предобученной модели ```MULAN trained on DeepLesion_epoch_08.pth```
([веса]:https://nihcc.app.box.com/s/agcs3orjctj981vyitwgrzaxcyx1qq71).
3. Для воспроизведения экспериментов с различными методами интерпретации результатов данной модели воспользуйтесь файлом ```object_detection_interpretation.ipynb```.
4. Для воспроизведения экспериментов с конкретным методом интерпретации для признаков, полученных в различных модулях модели, воспользуйтесь файлом ```features_comparison.ipynb```.
