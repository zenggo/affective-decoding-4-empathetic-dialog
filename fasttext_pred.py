import fasttext
import numpy as np

ftmodel = fasttext.FastText.load_model('./save/fasttext_empathetic_dialogues.mdl')

d = {}
d['context'] = np.load('empdial_dataset/sys_dialog_texts.test.npy', allow_pickle=True)
d['target'] = np.load('empdial_dataset/sys_target_texts.test.npy', allow_pickle=True)
d['emotion'] = np.load('empdial_dataset/sys_emotion_texts.test.npy', allow_pickle=True)

preds = []
for i in range(15):
    t = " </s> ".join(d['context'][i])
    pred, _ = ftmodel.predict(t, k=1)
    pred_emo = pred[0].split('__')[-1]
    preds.append(pred_emo)
    # print(t)
    # print(d['emotion'][i])
    # print(pred_emo)
    # print('---------')

np.save('./empdial_dataset/fasttest_pred_emotion_texts.test.npy', preds, allow_pickle=True)