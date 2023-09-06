FROM python:3.9

WORKDIR /app

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY . .

RUN pip install -r requirement.txt

ENV MPL BACKEND=Agg

CMD ["python", "play.py"]
