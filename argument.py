def dqn_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)  # 降低学习率
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=1.0, type=float)  # 更强的梯度裁剪
    
    # 学习率衰减参数
    parser.add_argument("--lr_decay", default=0.9995, type=float, help='learning rate decay factor')
    parser.add_argument("--lr_min", default=0.00005, type=float, help='minimum learning rate')
    parser.add_argument("--lr_patience", default=50, type=int, help='patience for learning rate reduction')
    parser.add_argument("--lr_factor", default=0.5, type=float, help='factor for learning rate reduction')
    
    # 收敛判断参数
    parser.add_argument("--convergence_window", default=30, type=int, help='window size for convergence check')
    parser.add_argument("--convergence_threshold", default=195, type=float, help='reward threshold for convergence')
    parser.add_argument("--convergence_check_interval", default=20, type=int, help='interval for convergence check')
    
    # DQN训练相关参数
    parser.add_argument("--batch_size", default=64, type=int, help='batch size for training')
    parser.add_argument("--memory_size", default=50000, type=int, help='replay buffer size')
    parser.add_argument("--update_target_freq", default=500, type=int, help='target network update frequency')
    parser.add_argument("--train_freq", default=2, type=int, help='training frequency')
    parser.add_argument("--epsilon_decay", default=0.9995, type=float, help='epsilon decay rate')

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(60000), type=int)  # 增加训练时间

    return parser


def pg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=16, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)

    return parser
