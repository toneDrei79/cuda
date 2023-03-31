#pragma once


void set_true_mats(float matL[3][3], float matR[3][3]);

void set_gray_mats(float matL[3][3], float matR[3][3]);

void set_color_mats(float matL[3][3], float matR[3][3]);

void set_halfcolor_mats(float matL[3][3], float matR[3][3]);

void set_optimized_mats(float matL[3][3], float matR[3][3]);

void set_anaglyph_mats(int choice, float mat_l[3][3], float mat_r[3][3]);