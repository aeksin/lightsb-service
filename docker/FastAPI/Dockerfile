FROM python:3.8-slim

COPY ./models/LightSB/ALAE/requirements.txt ./models/LightSB/ALAE/requirements.txt
COPY ./app/requirements.txt  ./app/requirements.txt 

RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq-dev libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r models/LightSB/ALAE/requirements.txt -r app/requirements.txt && \
    JACOBIAN_PATH=$(find /usr/local/lib/python3.8/site-packages/dlutils -name jacobian.py) && \
    sed -i '/from torch\.autograd\.gradcheck import zero_gradients/d' "$JACOBIAN_PATH" && \
    echo '\ndef zero_gradients(i):\n    for t in iter_gradients(i):\n        t.zero_()' >> "$JACOBIAN_PATH" && \
    pip install --no-cache-dir opencv-python ultralytics && \
    rm -rf ~/.cache/pip

COPY ./models ./models

ENV PYTHONPATH="./:./app:./models"

RUN python models/LightSB/ALAE/training_artifacts/download_all.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
