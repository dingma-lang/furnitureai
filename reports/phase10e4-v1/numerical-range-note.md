The formerly failing COCO photograph000000002347 now yields a finite unscaled
projection-weight derivative maximum of1337.869750977. Its corresponding FP32
value multiplied by the old instrumentation scale128 is171247.328125,
2.6143times FP16's65504maximum. The positive-loss projection
maximum on the same photograph is only0.020234389.

This is concrete observed-gradient evidence of a half-precision range hazard
at the projection-weight adjoint. An autocast linear operation casts its FP32
master weight to FP16;its native weight adjoint can overflow before conversion
back to the FP32 master tensor. The exact original E2 intermediate was not saved,
so this does not establish the earliest old overflow or its exact rounded value.
No scaled-GPU retry or alternate experiment was performed;the multiplication
above is CPU arithmetic on the recovered FP32 statistic. The primary report's
caution about the unknown original operation still applies.
