#include <stdio.h>
#include <stdlib.h>
#include "stdint.h"

int Screen_Edid_Test();

int main()
{
    int ret;
    printf("Hello world!\n");

    ret=Screen_Edid_Test();
    printf("ret %d\n",ret);

    return 0;
}

#define __DEBUG_SIM__

#ifdef __DEBUG_SIM__
#define _DEBUG_SIM printf
#else
#define _DEBUG_SIM(format,...)
#endif

typedef struct{
    uint32_t pixel_clock;
    uint16_t h_active;
    uint16_t v_active;
    uint16_t h_pulse_width;
    uint16_t v_pulse_width;
    uint16_t h_front_porch;
    uint16_t v_front_porch;
    uint16_t h_back_porch;
    uint16_t v_back_porch;
}str_create_edid;

uint8_t IIC_50H[256]={
    0X00,0Xff,0Xff,0Xff,0Xff,0Xff,0Xff,0X00,0X51,0Xb8,0X20,0X14,0X00,0X00,0X00,0X00,
    0X32,0X1e,0X01,0X04,0Xa5,0X1e,0X14,0X78,0X07,0Xee,0X95,0Xa3,0X54,0X4c,0X99,0X26,
    0X0f,0X50,0X54,0X00,0X00,0X00,0X01,0X01,0X01,0X01,0X01,0X01,0X01,0X01,0X01,0X01,
    0X01,0X01,0X01,0X01,0X01,0X01,0X44,0Xa3,0Xd8,0Xb4,0X90,0X90,0X28,0X60,0X30,0X20,
    0X86,0X04,0X2c,0Xc8,0X10,0X00,0X00,0X18,0X00,0X00,0X00,0X10,0X00,0X0a,0X20,0X20,
    0X20,0X20,0X20,0X20,0X20,0X20,0X20,0X20,0X20,0X20,0X00,0X00,0X00,0X10,0X00,0X0a,
    0X20,0X20,0X20,0X20,0X20,0X20,0X20,0X20,0X20,0X20,0X20,0X20,0X00,0X00,0X00,0Xfc,
    0X00,0X53,0X6B,0X79,0X43,0X6F,0X64,0X65,0X20,0X20,0X20,0X20,0X20,0X20,0X00,0X23,
};

uint8_t Screen_Edid_CheckCal(uint8_t *dat){
    uint8_t i;
    uint8_t sum=0;

    for(i=0;i<127;i++){
        sum+=dat[i];
    }

    return (256-sum);
}

void Screen_Edid_Cal_Res(str_create_edid r,uint8_t *dat){
    dat[0]=r.pixel_clock;//36
    dat[1]=r.pixel_clock>>8;//37

    dat[2]=r.h_active;//38
    dat[4]=(r.h_active>>4)&0xf0;//3a

    dat[5]=r.v_active;//3b
    dat[7]=(r.v_active>>4)&0xf0;//3d

    uint16_t hblank=r.h_pulse_width+r.h_front_porch+r.h_back_porch;
    uint16_t vblank=r.v_pulse_width+r.v_front_porch+r.v_back_porch;

    dat[3]=hblank;//39
    dat[6]|=(hblank>>8)&0x0f;//3d

    dat[6]=vblank;//3c
    dat[7]|=(vblank>>8)&0x0f;//3d

    dat[8]=r.h_front_porch;//3e
    dat[11]=(r.h_front_porch>>2)&0xc0;//41

    dat[9]=r.h_pulse_width;//3f
    dat[11]|=(r.h_pulse_width>>4)&0x30;//41

    dat[10]=r.v_front_porch<<4;//40
    dat[11]|=(r.v_front_porch>>2)&0x0c;//41

    dat[10]|=r.v_pulse_width&0x0f;//40
    dat[11]|=(r.v_pulse_width>>4)&0x03;//41

    uint16_t w_mm=300;
    dat[12]=w_mm;//42
    dat[14]=(w_mm>>4)&0xf0;//44

    uint16_t h_mm=200;
    dat[13]=h_mm;//43
    dat[14]|=(h_mm>>8)&0x0f;//44

    dat[15]=0x00;//45
    dat[16]=0x00;//46

    dat[17]=0x18;//47
}

int Screen_Edid_SetRes(str_create_edid res){
    Screen_Edid_Cal_Res(res,&IIC_50H[0x36]);
    IIC_50H[127]=Screen_Edid_CheckCal(IIC_50H);
    return 0;
}

int Screen_Edid_Test()
{
    str_create_edid resolving;

    resolving.pixel_clock=417.96*100;
    resolving.h_active=2520;
    resolving.v_active=1680;
    resolving.h_front_porch=48;
    resolving.h_pulse_width=32;
    resolving.h_back_porch=100;
    resolving.v_front_porch=24;
    resolving.v_pulse_width=6;
    resolving.v_back_porch=10;

    Screen_Edid_Cal_Res(resolving,&IIC_50H[0x36]);
    IIC_50H[127]=Screen_Edid_CheckCal(IIC_50H);

    for(uint8_t j=0;j<8;j++){
        for(uint8_t i=0;i<16;i++){
            _DEBUG_SIM("%02X ",IIC_50H[i+j*16]);

        }
        _DEBUG_SIM("\n");
    }
    _DEBUG_SIM("\n");
    //printf("%02X \n",);

    return 0;
}
