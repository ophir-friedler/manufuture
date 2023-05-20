import argparse
from project_template.utils.utils import *


def parse_args():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--query_date', type=str, required=True)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    # init prometheus and get metrics
    hit_counter, predicted_ctr_hist = get_metrics()

    # send counter inc to prometheus
    hit_counter.labels(type="hit").inc()

    # query vertica
    avg_final_predicted_ctr = get_avg_final_predicted_ctr(args.query_date)
    print('avg predicted ctr for {} is {}'.format(args.query_date, avg_final_predicted_ctr))

    # send hist to prometheus
    predicted_ctr_hist.labels(label1='label1_example', label2='label2_example').observe(avg_final_predicted_ctr)

    # send kibana event


if __name__ == '__main__':
    main()
