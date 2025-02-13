import os
import sys
import torch
import base64
from tqdm import tqdm
from data import SingleHandH5Dataset_MobileLastN
from cnn_2d import SimpleCNN2D, ComplexCNN2D

if __name__ == '__main__':
    model_file = sys.argv[1]
    print(f"Running {model_file}")

    is_lstm = False
    window_length = 60
    batch_size = 2 ** 0

    train_loader = torch.utils.data.DataLoader(
        SingleHandH5Dataset_MobileLastN("/data/train", "/meta/563_sign_list.txt", debug=True, n=window_length, is_lstm=is_lstm),
        batch_size=batch_size, shuffle=False,
        num_workers=16
    )
    val_loader = torch.utils.data.DataLoader(
        SingleHandH5Dataset_MobileLastN("/data/validation", "/meta/563_sign_list.txt", debug=True, n=window_length, is_lstm=is_lstm),
        batch_size=batch_size, shuffle=False,
        num_workers=16
    )
    test_loader = torch.utils.data.DataLoader(
        SingleHandH5Dataset_MobileLastN("/data/test", "/meta/563_sign_list.txt", debug=True, n=window_length, is_lstm=is_lstm),
        batch_size=batch_size, shuffle=False,
        num_workers=16
    )

    with open("/meta/563_sign_list.txt") as f:
        label_order = [i.strip() for i in f.readlines()]

    sets = {
        "test": test_loader,
        "train": train_loader,
        "validation": val_loader
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # model = ComplexCNN2D(window_length, 21, 2, 563)
    # model.load_state_dict(torch.load(f"/models/{model_file}.pth"))
    # model.to(device)
    # model.eval()
    # model = model.to(device)

    # model.eval()

    # print(model)
    model = torch.load(f"/models/{model_file}.pt").to(device).eval()

    set_corrects = {}
    set_totals = {}

    for i in sets:
        set_corrects[i] = 0
        set_totals[i] = 0
        correct = {}
        total = {}
        print(f"=====Starting {i} set=====")
        sign_files = {}
        if not os.path.exists(f"/reports/{model_file}/{i}"):
            os.makedirs(f"/reports/{model_file}/{i}")
        
        for sign in label_order:
            sign_files[sign] = open(f"/reports/{model_file}/{i}/{sign}.csv", "w")
            sign_files[sign].write(f'"file", "sign", "guessed_sign", "predictions"\n')
            correct[sign] = 0
            total[sign] = 0

        with torch.no_grad():
            for input, label, file in tqdm(sets[i]):
                file = os.path.basename(file[0])
                sign = label_order[label.argmax()]
                input = input.to(device)
                label = label.to(device)
                outs = model(input)
                guessed_sign = label_order[torch.softmax(outs, dim=1).argmax(dim=1).item()]
                sign_files[sign].write('"' + file.replace("\"", "\\\"") + f'", "{sign}", "{guessed_sign}", "{base64.b64encode(str(outs.cpu().detach().numpy()[0].tolist()).encode("utf-8"))}"' + '\n')
                correct[sign] += 1 if guessed_sign == sign else 0
                total[sign] += 1

                # correct[sign] += (label.argmax(dim=1) == torch.softmax(outs, dim=1).argmax(dim=1)).sum().item()
                # total[sign] += label.size(0)
        
        for sign in label_order:
            sign_files[sign].close()

        with open(f"/reports/{model_file}/{i}_summary.csv", "w") as f:
            f.write(f'"sign", "correct", "total", "accuracy"\n')
            for sign in label_order:
                f.write(f'"{sign}", {correct[sign]}, {total[sign]}, {correct[sign]/ total[sign] if total[sign] > 0 else "None"}\n')
                set_corrects[i] += correct[sign]
                set_totals[i] += total[sign]

    with open(f"/reports/{model_file}/summary.csv", "w") as f:
        f.write(f'"set", "correct", "total", "accuracy"\n')
        for i in sets:
            f.write(f'"{i}", {set_corrects[i]}, {set_totals[i]}, {set_corrects[i] / set_totals[i]}\n')


            

    
