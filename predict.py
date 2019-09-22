
from utils import parse_predict_args, device, print_predict_result
from model import load_check_point, build_nllloss_criterion, predict

# Parse command-line arguments
args = parse_predict_args()

# Device
device = device(args.gpu)

# Rebuild model and optimizer from check point
model, optimizer = load_check_point(args.check_point_file, device)
criterion = build_nllloss_criterion()

top_ps, top_classes = predict(image_file=args.image_file, model=model, device=device, topk=args.topk)
print_predict_result(top_ps, top_classes, args.cat_to_name_file)


