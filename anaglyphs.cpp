#include "anaglyphs.h"


const float trueL[3][3] = {
    {0.299, 0.587, 0.114},
    {0.   , 0.   , 0.   },
    {0.   , 0.   , 0.   }
};
const float trueR[3][3] = {
    {0.   , 0.   , 0.   },
    {0.   , 0.   , 0.   },
    {0.299, 0.587, 0.114}
};

const float grayL[3][3] = {
    {0.299, 0.587, 0.114},
    {0.   , 0.   , 0.   },
    {0.   , 0.   , 0.   }
};
const float grayR[3][3] = {
    {0.   , 0.   , 0.   },
    {0.299, 0.587, 0.114},
    {0.299, 0.587, 0.114}
};

const float colorL[3][3] = {
    {1.   , 0.   , 0.   },
    {0.   , 0.   , 0.   },
    {0.   , 0.   , 0.   }
};
const float colorR[3][3] = {
    {0.   , 0.   , 0.   },
    {0.   , 1.   , 0.   },
    {0.   , 0.   , 1.   }
};

const float halfcolorL[3][3] = {
    {0.299, 0.587, 0.114},
    {0.   , 0.   , 0.   },
    {0.   , 0.   , 0.   }
};
const float halfcolorR[3][3] = {
    {0.   , 0.   , 0.   },
    {0.   , 1.   , 0.   },
    {0.   , 0.   , 1.   }
};

const float optimizedL[3][3] = {
    {0.   , 0.7  , 0.3  },
    {0.   , 0.   , 0.   },
    {0.   , 0.   , 0.   }
};
const float optimizedR[3][3] = {
    {0.   , 0.   , 0.   },
    {0.   , 1.   , 0.   },
    {0.   , 0.   , 1.   }
};


void setTrueAnaglyphMats(float matL[3][3], float matR[3][3])
{
    for (int j=0; j<3; j++)
    {
        for (int i=0; i<3; i++)
        {
            matL[j][i] = trueL[j][i];
            matR[j][i] = trueR[j][i];
        }
    }
}

void setGrayAnaglyphMats(float matL[3][3], float matR[3][3])
{
    for (int j=0; j<3; j++)
    {
        for (int i=0; i<3; i++)
        {
            matL[j][i] = grayL[j][i];
            matR[j][i] = grayR[j][i];
        }
    }
}

void setColorAnaglyphMats(float matL[3][3], float matR[3][3])
{
    for (int j=0; j<3; j++)
    {
        for (int i=0; i<3; i++)
        {
            matL[j][i] = colorL[j][i];
            matR[j][i] = colorR[j][i];
        }
    }
}

void setHalfcolorAnaglyphMats(float matL[3][3], float matR[3][3])
{
    for (int j=0; j<3; j++)
    {
        for (int i=0; i<3; i++)
        {
            matL[j][i] = halfcolorL[j][i];
            matR[j][i] = halfcolorR[j][i];
        }
    }
}

void setOptimizedAnaglyphMats(float matL[3][3], float matR[3][3])
{
    for (int j=0; j<3; j++)
    {
        for (int i=0; i<3; i++)
        {
            matL[j][i] = optimizedL[j][i];
            matR[j][i] = optimizedR[j][i];
        }
    }
}