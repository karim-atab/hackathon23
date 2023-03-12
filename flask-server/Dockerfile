FROM python:3.9
RUN mkdir /server
WORKDIR /server
COPY requirements.txt new-com-data.csv server.py categorymodel.sav ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
ENV FLASK_ENV server.py

EXPOSE 5000
CMD ["python","server.py"]