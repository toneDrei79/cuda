#include "anaglyph_mats.h"


const float true_l[3][3] = {
    {0.299, 0.587, 0.114},
    {0.   , 0.   , 0.   },
    {0.   , 0.   , 0.   }
};
const float true_r[3][3] = 
{
    {0.   , 0.   , 0.   },
    {0.   , 0.   , 0.   },
    {0.299, 0.587, 0.114}
};

const float gray_l[3][3] = {
    {0.299, 0.587, 0.114},
    {0.   , 0.   , 0.   },
    {0.   , 0.   , 0.   }
};
const float gray_r[3][3] = {
    {0.   , 0.   , 0.   },
    {0.299, 0.587, 0.114},
    {0.299, 0.587, 0.114}
};

const float color_l[3][3] = {
    {1.   , 0.   , 0.   },
    {0.   , 0.   , 0.   },
    {0.   , 0.   , 0.   }
};
const float color_r[3][3] = {
    {0.   , 0.   , 0.   },
    {0.   , 1.   , 0.   },
    {0.   , 0.   , 1.   }
};

const float halfcolor_l[3][3] = {
    {0.299, 0.587, 0.114},
    {0.   , 0.   , 0.   },
    {0.   , 0.   , 0.   }
};
const float halfcolor_r[3][3] = {
    {0.   , 0.   , 0.   },
    {0.   , 1.   , 0.   },
    {0.   , 0.   , 1.   }
};

const float optimized_l[3][3] = {
    {0.   , 0.7  , 0.3  },
    {0.   , 0.   , 0.   },
    {0.   , 0.   , 0.   }
};
const float optimized_r[3][3] = {
    {0.   , 0.   , 0.   },
    {0.   , 1.   , 0.   },
    {0.   , 0.   , 1.   }
};


void set_true_mats(float mat_l[3][3], float mat_r[3][3])
{
    for (int j=0; j<3; j++)
    {
        for (int i=0; i<3; i++)
        {
            mat_l[j][i] = true_l[j][i];
            mat_r[j][i] = true_r[j][i];
        }
    }
}

void set_gray_mats(float mat_l[3][3], float mat_r[3][3])
{
    for (int j=0; j<3; j++)
    {
        for (int i=0; i<3; i++)
        {
            mat_l[j][i] = gray_l[j][i];
            mat_r[j][i] = gray_r[j][i];
        }
    }
}

void set_color_mats(float mat_l[3][3], float mat_r[3][3])
{
    for (int j=0; j<3; j++)
    {
        for (int i=0; i<3; i++)
        {
            mat_l[j][i] = color_l[j][i];
            mat_r[j][i] = color_r[j][i];
        }
    }
}

void set_halfcolor_mats(float mat_l[3][3], float mat_r[3][3])
{
    for (int j=0; j<3; j++)
    {
        for (int i=0; i<3; i++)
        {
            mat_l[j][i] = halfcolor_l[j][i];
            mat_r[j][i] = halfcolor_r[j][i];
        }
    }
}

void set_optimized_mats(float mat_l[3][3], float mat_r[3][3])
{
    for (int j=0; j<3; j++)
    {
        for (int i=0; i<3; i++)
        {
            mat_l[j][i] = optimized_l[j][i];
            mat_r[j][i] = optimized_r[j][i];
        }
    }
}


void set_anaglyph_mats(int choice, float mat_l[3][3], float mat_r[3][3])
{
    switch (choice)
    {
        case 0: set_true_mats(mat_l, mat_r); break;
        case 1: set_gray_mats(mat_l, mat_r); break;
        case 2: set_color_mats(mat_l, mat_r); break;
        case 3: set_halfcolor_mats(mat_l, mat_r); break;
        case 4: set_optimized_mats(mat_l, mat_r); break;
        default:;
    }
}