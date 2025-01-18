# news-category-prediction

This is a google cloud endpoint repository for news category prediction

Endpoint URL: https://predict-category-750241619106.us-east1.run.app

**Example**:

`Article`: https://www.bbc.com/travel/article/20250108-museum-manchester-trend-attention-span

`Request`: 


https://predict-category-750241619106.us-east1.run.app/predict/?headline=The new museum trend helping us regain our lost attention&link=https://www.bbc.com/travel/article/20250108-museum-manchester-trend-attention-span&short_description=In a Manchester art gallery, a quiet room with just three paintings is starting a mental health movement aimed at reclaiming our lost attention spans.There's a small, dark green room in Manchester Art Gallery, right next to a packed gallery of L S Lowry's stick figures and factory buildings. While people mill around in the bright space, as busy as the artist's workers outside the city's factories, in the dark green room they slow right down. They sit and look at the three paintings on the wall and really see them. Guided by a downloadable meditation, visitors are encouraged to spend up to 15 minutes with their chosen artwork, one on one.&authors=Laura Hall

`Output`: Prediction for this article: TRAVEL