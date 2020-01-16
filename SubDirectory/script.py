import pandas as pd
import numpy as np
import keras
import tensorflow


def snake(var):
    cols = ['X.HR.', 'X.RESP.', 'X.SpO2.', 'final_abp_sys', 'final_abp_dias']
    data = var
    hr = np.array(data[[cols[0]]]).reshape(-1, 256)
    rr = np.array(data[[cols[1]]]).reshape(-1, 256)
    spo = np.array(data[[cols[2]]]).reshape(-1, 256)
    sys = np.array(data[[cols[3]]]).reshape(-1, 256)
    dia = np.array(data[[cols[4]]]).reshape(-1, 256)

    # applying snake
    X_dia_snake = np.zeros((dia.shape[0], 256), dtype=np.float32)
    X_hr_snake = np.zeros((hr.shape[0], 256), dtype=np.float32)
    X_rr_snake = np.zeros((rr.shape[0], 256), dtype=np.float32)
    X_spo_snake = np.zeros((spo.shape[0], 256), dtype=np.float32)
    X_sys_snake = np.zeros((sys.shape[0], 256), dtype=np.float32)

    for i in range(dia.shape[0]):
        b = dia[i]
        k = 0
        print('on audio {}'.format(i))
        for j in range(0, dia.shape[1], 16):
            if k % 2 == 0:
                X_dia_snake[i][j:j + 16] = b[j:j + 16]
            else:
                X_dia_snake[i][j:j + 16] = np.flip(b[j:j + 16], 0)
            k += 1

    for i in range(hr.shape[0]):
        b = hr[i]
        k = 0
        print('on audio {}'.format(i))
        for j in range(0, hr.shape[1], 16):
            if k % 2 == 0:
                X_hr_snake[i][j:j + 16] = b[j:j + 16]
            else:
                X_hr_snake[i][j:j + 16] = np.flip(b[j:j + 16], 0)
            k += 1

    for i in range(rr.shape[0]):
        b = rr[i]
        k = 0
        print('on audio {}'.format(i))
        for j in range(0, rr.shape[1], 16):
            if k % 2 == 0:
                X_rr_snake[i][j:j + 16] = b[j:j + 16]
            else:
                X_rr_snake[i][j:j + 16] = np.flip(b[j:j + 16], 0)
            k += 1

    for i in range(spo.shape[0]):
        b = spo[i]
        k = 0
        print('on audio {}'.format(i))
        for j in range(0, spo.shape[1], 16):
            if k % 2 == 0:
                X_spo_snake[i][j:j + 16] = b[j:j + 16]
            else:
                X_spo_snake[i][j:j + 16] = np.flip(b[j:j + 16], 0)
            k += 1

    for i in range(sys.shape[0]):
        b = sys[i]
        k = 0
        print('on audio {}'.format(i))
        for j in range(0, sys.shape[1], 16):
            if k % 2 == 0:
                X_sys_snake[i][j:j + 16] = b[j:j + 16]
            else:
                X_sys_snake[i][j:j + 16] = np.flip(b[j:j + 16], 0)
            k += 1

    x = np.dstack([X_dia_snake, X_hr_snake, X_rr_snake, X_spo_snake, X_sys_snake])
    x = x.reshape(-1, 16, 16, 5)

    return x