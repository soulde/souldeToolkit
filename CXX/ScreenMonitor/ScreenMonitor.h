//
// Created by lyjly on 2023/1/18.
//

#ifndef SOULDETOOLKIT_SCREENMONITER_H
#define SOULDETOOLKIT_SCREENMONITER_H

#include <windows.h>

class ScreenMonitor {
public:
    ScreenMonitor();
    void read();
    ~ScreenMonitor();

private:
    HWND handler;
    HDC hWindowDC;
    HDC hWindowCompatibleDC;
    HBITMAP hbWindow;
};


#endif //SOULDETOOLKIT_SCREENMONITER_H
