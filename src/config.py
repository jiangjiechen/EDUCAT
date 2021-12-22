import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)

    parser.add_argument('--pool_size', '-p', type=int, default=1, help='Multiprocessing')
    parser.add_argument('--rank_0', type=int, default=1, choices=[0, 1])
    parser.add_argument('--device', default=None)
    parser.add_argument('--drop_time', type=int, default=60, help='Burn-in: 60% of sample time')
    parser.add_argument('--min_length', type=int, default=2)
    parser.add_argument('--sample_time', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--search_size', type=int, default=100, help='Number of proposed tokens in mlm')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--coherence_type', '-c', choices=['lm', 'ents', 'null'], default='null')

    parser.add_argument('--causal_token_finding', action='store_true', help='Conflict detection')
    parser.add_argument('--abduction', action='store_true', help='Add coherence constraint')
    parser.add_argument('--action_threshold', type=float, default=1.3, help='Action proposal threshold')
    parser.add_argument('--annealing_parameter', type=float, default=0.95,
                        help='Annealing parameter for faster convergence to optimal, 1 for not annealing.')

    parser.add_argument('--just_acc_rate', type=float, default=0.0)
    parser.add_argument('--restrict_constr', action='store_true')
    parser.add_argument('--action_prob', nargs='+', type=float, default=[0.34, 0.33, 0.33],
                        help='replace, insert and delete')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--mlm_path', type=str, default='../models/roberta-base',
                        help='Currently support RoBERTa for tokenization consistency with GPT2')
    parser.add_argument('--constraint_model_path', type=str,
                        help='Path to BERT or fine-tuned BERT or EntScore')
    parser.add_argument('--gpt2_path', type=str, default='../models/gpt2-medium')

    parser.add_argument('--lark', action='store_true')
    args = parser.parse_args()
    return args
