import numpy as np


def load_results(
    results_file,
    label=None,
    MAX_IOU=90,
    dataset=None,
    dataset_classes=None,
    exclude_classes=["wall", "ceiling", "floor", "unlabelled", "unlabeled"],
    dataset_name="semKITTI",
):
    objects = {}

    def process_objects(data):
        nonlocal objects
        for entry in data:
            objects[entry[0].replace("scene", "") + "_" + entry[1]] = 1

    def filter_objects_by_classes(dataset_, dataset_classes, exclude_classes):

        mask = np.isin(dataset_classes, exclude_classes, invert=True)
        print("total number of objects: ", np.shape(dataset_classes))
        print("number of objects kept: ", sum(mask))
        return dataset_[mask], dataset_classes[mask]

    if exclude_classes:
        dataset, dataset_classes = filter_objects_by_classes(
            dataset, dataset_classes, exclude_classes
        )
        process_objects(dataset)
    if label is not None:
        dataset = dataset[dataset_classes == label]
        objects = {}
        process_objects(dataset)
        print("number of objects kept from class ", label, ": ", len(objects))
    else:
        process_objects(dataset)
        print("number of objects kept: ", len(objects))

    results_dict_KatIOU = {}
    num_objects = 0
    ordered_clicks = []
    all_object = {}
    results_dict_per_click = {}
    results_dict_per_click_iou = {}
    all_data = {}
    corrupted = 0

    with open(results_file, "r") as f:
        for line in f:
            splits = line.rstrip().split(" ")
            scene_name = splits[1].replace("scene", "")
            object_id = splits[2]
            num_clicks = splits[3]
            iou = splits[4]

            obj_key = scene_name + "_" + object_id

            if obj_key in objects:
                all_object.setdefault(obj_key, 1)
                all_data.setdefault(obj_key, []).append((num_clicks, iou))

                if float(iou) >= MAX_IOU:
                    if obj_key not in results_dict_KatIOU:
                        results_dict_KatIOU[obj_key] = float(num_clicks)
                        num_objects += 1
                        ordered_clicks.append(float(num_clicks))
                elif int(num_clicks) >= 20 and float(iou) >= 0:
                    if float(iou) == 0 and dataset_name == "semKITTI":
                        corrupted += 1  # no enough object instance points
                    else:
                        if obj_key not in results_dict_KatIOU:
                            results_dict_KatIOU[obj_key] = float(num_clicks)
                            num_objects += 1
                            ordered_clicks.append(float(num_clicks))

                results_dict_per_click.setdefault(num_clicks, 0)
                results_dict_per_click_iou.setdefault(num_clicks, 0)

                results_dict_per_click[num_clicks] += 1
                results_dict_per_click_iou[num_clicks] += float(iou)

    if not results_dict_KatIOU:
        print("no objects to evaluate")
        return 0

    click_at_80 = sum(results_dict_KatIOU.values()) / len(results_dict_KatIOU.values())
    print(
        "click@",
        MAX_IOU,
        click_at_80,
        num_objects,
        len(all_object),
        len(results_dict_KatIOU),
    )
    clicks = [str(i) for i in range(21)]
    iou = [results_dict_per_click_iou[i] / results_dict_per_click[i] for i in clicks]

    for obj_key, data in all_data.items():
        if len(data) != 21:
            pass  # Handle incomplete data if needed

    return ordered_clicks, clicks, iou


if __name__ == "__main__":
    datasets = ["semKITTI"]

    if "semKITTI" in datasets:
        path = path = "./results/semanticKITTI/"

        base_005_005 = "semKITTI_baseline_all2_005_005_all_14.csv"
        dataset_semKITTI = np.load("./results/semanticKITTI/dataset_01.npy")
        dataset_classes_semKITTI = np.loadtxt(
            "./results/semanticKITTI/dataset_01_classes.txt", dtype=str
        )
        print("SemanticKITTI")
        for iou_max in [80, 85, 90]:
            print("++++")
            print("base")
            print("005 to 005 - excluded unlabeled")
            load_results(
                path + base_005_005,
                None,
                iou_max,
                dataset_semKITTI,
                dataset_classes_semKITTI,
                None,
                "semKITTI",
            )
