import numpy as np

def validate_1hot_outputs(oh_labels):
    oh_labels = np.array(oh_labels)

    assert oh_labels[0] is not None, 'One Hot Output Labels are None'

    base = oh_labels[0].reshape(-1)

    for i in range(1, oh_labels.shape[0]):
        label = oh_labels[i].reshape(-1)
        for j in range(label.shape[0]):
            assert label[j] == base[j], 'One Hot Output Labels are different'