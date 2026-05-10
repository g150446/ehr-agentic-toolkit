#import "TemplateMatchingWrapper.h"
#import <opencv2/core.hpp>
#import <opencv2/imgproc.hpp>

@implementation TemplateMatchResult @end

@implementation TemplateMatchingWrapper

static cv::Mat cgImageToMat(CGImageRef img) {
    size_t w = CGImageGetWidth(img), h = CGImageGetHeight(img);
    cv::Mat rgba((int)h, (int)w, CV_8UC4);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        rgba.data, w, h, 8, rgba.step[0], cs,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), img);
    CGContextRelease(ctx);
    CGColorSpaceRelease(cs);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    return bgr;
}

+ (nullable TemplateMatchResult *)matchSource:(CGImageRef)source
                                     template:(CGImageRef)tmpl
                                 searchRegion:(CGRect)region
                                    threshold:(double)threshold {
    cv::Mat srcMat = cgImageToMat(source);
    cv::Mat tplMat = cgImageToMat(tmpl);

    int rx = MAX(0, (int)region.origin.x);
    int ry = MAX(0, (int)region.origin.y);
    int rw = MIN((int)region.size.width,  srcMat.cols - rx);
    int rh = MIN((int)region.size.height, srcMat.rows - ry);
    if (rw <= tplMat.cols || rh <= tplMat.rows) return nil;

    cv::Mat roi = srcMat(cv::Rect(rx, ry, rw, rh));
    cv::Mat result;
    cv::matchTemplate(roi, tplMat, result, cv::TM_CCOEFF_NORMED);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    if (maxVal < threshold) return nil;

    TemplateMatchResult *res = [TemplateMatchResult new];
    res.position = CGPointMake(rx + maxLoc.x, ry + maxLoc.y);
    res.score = maxVal;
    return res;
}

@end
