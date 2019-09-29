FROM python:3

WORKDIR /usr/src/app

ADD requirement.txt /usr/src/app

RUN pip install -r requirement.txt
RUN mkdir workspace

CMD [ "bash" ]

