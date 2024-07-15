FROM registry.access.redhat.com/ubi9/python-311:latest as builder
WORKDIR /dataset
RUN pip install datasets

RUN curl https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data.json --output alpaca_data.json
RUN curl https://raw.githubusercontent.com/openshift-psap/topsail/a513e4530d56de36c713416690d8edb117588f89/projects/fine_tuning/roles/fine_tuning_run_fine_tuning_job/files/entrypoint/convert_replicate.py --output convert_replicate.py
COPY alpaca_format.py alpaca_format.py

RUN python alpaca_format.py alpaca_data.json alpaca_data_formatted.json
RUN python convert_replicate.py alpaca_data_formatted.json alpaca_data_half.json 0.5
RUN python convert_replicate.py alpaca_data_formatted.json alpaca_data_quarter.json 0.25
RUN python convert_replicate.py alpaca_data_formatted.json alpaca_data_tenth.json 0.1
RUN python convert_replicate.py alpaca_data_formatted.json alpaca_data_hundredth.json 0.01

FROM registry.access.redhat.com/ubi9:latest
WORKDIR /dataset
RUN curl https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/DATA_LICENSE --output DATA_LICENSE
RUN curl https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/README.md --output README.md

COPY --from=builder /dataset/alpaca_data.json alpaca_data.json
COPY --from=builder /dataset/alpaca_data_half.json alpaca_data_half.json
COPY --from=builder /dataset/alpaca_data_quarter.json alpaca_data_quarter.json
COPY --from=builder /dataset/alpaca_data_tenth.json alpaca_data_tenth.json
COPY --from=builder /dataset/alpaca_data_hundredth.json alpaca_data_hundredth.json
