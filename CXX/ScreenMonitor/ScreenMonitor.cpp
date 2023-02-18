//
// Created by lyjly on 2023/1/18.
//

#include "ScreenMonitor.h"

ScreenMonitor::ScreenMonitor() {
    handler = GetDesktopWindow();
    hWindowDC= GetDC(handler);
    hWindowCompatibleDC = CreateCompatibleDC(hWindowDC);
    SetStretchBltMode(hWindowCompatibleDC, COLORONCOLOR);
}

ScreenMonitor::~ScreenMonitor() {
    DeleteObject(hbWindow);
    DeleteDC(hWindowCompatibleDC);
    ReleaseDC(handler, hWindowDC);
}

void ScreenMonitor::read() {

    // define scale, height and width
    int screenX = GetSystemMetrics(SM_XVIRTUALSCREEN);
    int screenY = GetSystemMetrics(SM_YVIRTUALSCREEN);
    int width = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    int height = GetSystemMetrics(SM_CYVIRTUALSCREEN);

    // create a bitmap
    hbWindow = CreateCompatibleBitmap(hWindowDC, width, height);
    BITMAPINFOHEADER bi;
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height;  //this is the line that makes it draw upside down or not
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;
    bi.biXPelsPerMeter = 0;
    bi.biYPelsPerMeter = 0;
    bi.biClrUsed = 0;
    bi.biClrImportant = 0;

    // use the previously created device context with the bitmap
    SelectObject(hWindowCompatibleDC, hbWindow);

    // copy from the window device context to the bitmap device context
    StretchBlt(hWindowCompatibleDC, 0, 0, width, height, hWindowDC, screenX, screenX, width, height, SRCCOPY);  //change SRCCOPY to NOTSRCCOPY for wacky colors !
    GetDIBits(hWindowCompatibleDC, hbWindow, 0, height, src.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);            //copy from hwindowCompatibleDC to hbWindow

}
