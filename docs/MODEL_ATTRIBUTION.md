# Model Attribution

This project uses a pre-trained model from HuggingFace. We did not train this model.

## Model Information

| Property | Value |
|----------|-------|
| **Model Name** | Gym-Workout-Classifier-SigLIP2 |
| **Author** | prithivMLmods |
| **HuggingFace URL** | https://huggingface.co/prithivMLmods/Gym-Workout-Classifier-SigLIP2 |
| **DOI** | 10.57967/hf/5391 |
| **Base Architecture** | google/siglip2-base-patch16-224 |

## Performance Metrics

The following metrics are **reported by the original model authors**, not independently validated by us:

| Metric | Value |
|--------|-------|
| Accuracy | 96.38% |
| Test Images | 9,687 |
| Classes | 22 |

## Supported Exercise Classes

The model recognizes 22 gym exercises:

1. barbell biceps curl
2. bench press
3. chest fly machine
4. deadlift
5. decline bench press
6. hammer curl
7. hip thrust
8. incline bench press
9. lat pulldown
10. lateral raises
11. leg extension
12. leg raises
13. plank
14. pull up
15. push up
16. romanian deadlift
17. russian twist
18. shoulder press
19. squat
20. t bar row
21. tricep dips
22. tricep pushdown

## Our Contributions

Our work focuses on the application layer, not model training:

1. **Real-time Integration** - Webcam video processing with OpenCV
2. **Temporal Smoothing** - 3-frame consensus mechanism to reduce false positives during exercise transitions
3. **Motion Detection** - Frame differencing approach for repetition counting (proposed)
4. **Analytics Dashboard** - Web-based visualization of workout history

## Citation

If you use this model, please cite the original authors:

```
@misc{prithivMLmods2024gym,
  author = {prithivMLmods},
  title = {Gym-Workout-Classifier-SigLIP2},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/prithivMLmods/Gym-Workout-Classifier-SigLIP2},
  doi = {10.57967/hf/5391}
}
```

## License

The model is hosted on HuggingFace. Please refer to the [model page](https://huggingface.co/prithivMLmods/Gym-Workout-Classifier-SigLIP2) for licensing information.
