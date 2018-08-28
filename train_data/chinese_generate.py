#!/usr/bin/python
#encoding: utf-8
import os
import pygame

chinese_dir = 'chinese'
if not os.path.exists(chinese_dir):
        os.mkdir(chinese_dir)

pygame.init()

for head in range(int(0xb0), int(0xf7) + 1):
    for body in range(int(0xa1), int(0xfe) + 1):
        val = '{:x}{:x}'.format(head, body)
        word = val.decode('hex').decode('gb2312', 'ignore')
        font = pygame.font.Font("KaiTi.ttf", 128)
        rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
        pygame.image.save(rtext, os.path.join(chinese_dir, word + ".bmp"))
