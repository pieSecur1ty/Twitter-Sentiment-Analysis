# Twitter Sentiment Analysis



## 1. Import Necessary Modules
    import pandas as pd
    import neattext.functions as nfx
    from wordcloud import WordCloud 
    import matplotlib.pyplot as plt
    from textblob import TextBlob
    

## 2. Load Dataset
    dataset = pd.read_csv("text_emotion.csv", encoding="latin-1")
    df = pd.DataFrame(dataset)
    df.head()
    
   ![image](https://user-images.githubusercontent.com/103514216/163519616-717bb318-9d3b-4160-92bd-9824e4bb4655.png)


## 3. Clean The Dataset
    df['content'] = df['content'].apply(nfx.remove_hashtags) # remove hashtags
    df['content'] = df['content'].apply(nfx.remove_urls) # remove urls
    df['content'] = df['content'].apply(nfx.remove_emojis) # remove emojies
    df['content'] = df['content'].apply(nfx.remove_userhandles) # remove userhandles
    df['content'] = df['content'].apply(nfx.remove_dates) # remove dates
    df['content'] = df['content'].apply(nfx.remove_punctuations) # remove punctuations
    df.head()
    
   ![image](https://user-images.githubusercontent.com/103514216/163520573-29a2cd74-3e8d-43f5-9e45-9db416214651.png)


## 4. Plot The Wordcloud
    all_words = ' '.join(df['content'])
    word_cloud = WordCloud(height = 1000, width = 1000, random_state = 42, max_font_size = 220).generate(all_words)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
   ![image](https://user-images.githubusercontent.com/103514216/163520954-75b3be2c-8158-4d05-8006-ac5e3d2b898a.png)


## 5. Calculate Subjectivity
    def get_subjectivity(text):
        return TextBlob(text).sentiment.subjectivity
    df['subjectivity'] = df['content'].apply(get_subjectivity)

    df.head()
    
   ![image](https://user-images.githubusercontent.com/103514216/163521375-85cef7dc-5704-471f-bd7d-6847ee060652.png)


## 6. Calculate Polarity
    def get_polarity(text):
        return TextBlob(text).sentiment.polarity

    df['polarity'] = df['content'].apply(get_polarity)
    df.head()
    
   ![image](https://user-images.githubusercontent.com/103514216/163521721-affa49d8-4288-459b-a7fd-bc215f0357bf.png)


## 7. Plot Scatter Graph Of Polarity v/s Subjectivity
    plt.figure(figsize=(15,5))
    plt.title("\n[ Polarity v/s Subjectivity ]\n")
    plt.xlabel("\nPolarity\n")
    plt.ylabel("\nSubjectivity\n")
    plt.scatter(df['polarity'], df['subjectivity'], color='blue')
    plt.show()
    
  ![image](https://user-images.githubusercontent.com/103514216/163522255-30d98354-3f43-4849-a4d5-561a2e0b0050.png)


## 8. Analysis The Sentiment On The Basis Of Polarity
    def get_sentiment(score):
        if score > 0:
            return 'positive'
        elif score < 0:
            return 'negative'
        else:
            return 'neutral'

    df['analysis'] = df['polarity'].apply(get_sentiment)
    df.head()
    
   ![image](https://user-images.githubusercontent.com/103514216/163523114-e6994877-2373-45c9-abc1-88620ae8aa68.png)


## 9. Calculate Positive, Negative And Neutral Tweets Percentage
    values = df['analysis'].value_counts()
    positive_tweet_percentage = round((values["positive"] / df['analysis'].shape[0]) * 100)
    negative_tweet_percentage = round((values["negative"] / df['analysis'].shape[0]) * 100)
    neutral_tweet_percentage = round((values["neutral"] / df['analysis'].shape[0]) * 100)

    tweets_percentage = [positive_tweet_percentage, negative_tweet_percentage, neutral_tweet_percentage]
    tweets_percentage
    
   [44, 21, 35]
   
 
## 10. Plot Bar Graph Of Analysis And Visualization
    plt.figure(figsize=(15, 5))
    plt.title("\n[ Sentiment Analysis ]\n")
    plt.xlabel("\nAnalysis\n")
    plt.ylabel("\nCounts\n")
    df['analysis'].value_counts().plot(kind='bar', color=["Green", "Purple", "Blue"])
    plt.show()
    
  ![image](https://user-images.githubusercontent.com/103514216/163523593-2f09daef-8c64-4d29-8d93-bc7d4d1ab68d.png)



## Note :
   1. The code is written in 'Jupyter Lab'
   2. To install necessary modules in Debian/Ubuntu Based Operating System Terminal
   
     apt update
     apt install python3-pip
     pip install pandas matplotlib neattext textblob wordcloud

