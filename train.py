from utils import parse_train_args, load_data, device
from model import build_model, build_adam_optimizer, build_nllloss_criterion, train_model, save_check_point

# Parse command-line arguments
args = parse_train_args()

# Train and validation data loaders
train_data_loader = load_data(args.data_dir, 'train')
validation_data_loader = load_data(args.data_dir, 'valid')

# Device
device = device(args.gpu)

# Build model, optimizer, criterion
model = build_model(arch=args.arch, hidden_units=args.hidden_units, device=device)
optimizer = build_adam_optimizer(model, learning_rate=args.learning_rate)
criterion = build_nllloss_criterion()

# Train model, evaluate and print result
train_model(model, criterion, optimizer, args.epochs, train_data_loader, validation_data_loader, device, 40)

# Save check point of the model
check_point_file_name_prefix = 'check_point'
check_point_file_ext = '.pth'
save_check_point(model, optimizer, args.save_dir + '/' + check_point_file_name_prefix + '_' + args.arch + check_point_file_ext)


