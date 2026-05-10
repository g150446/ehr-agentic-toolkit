#pragma once
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

@interface TemplateMatchResult : NSObject
@property (nonatomic) CGPoint position;  // top-left of matched region in source image coords
@property (nonatomic) double score;      // TM_CCOEFF_NORMED score [0,1], 1.0 = perfect match
@end

@interface TemplateMatchingWrapper : NSObject
/// Returns nil if best match score < threshold.
+ (nullable TemplateMatchResult *)matchSource:(CGImageRef _Nonnull)source
                                     template:(CGImageRef _Nonnull)tmpl
                                 searchRegion:(CGRect)region
                                    threshold:(double)threshold;
@end
