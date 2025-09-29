"""
Data generation utilities for creating realistic sample news datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

class NewsDataGenerator:
    """Generate synthetic news articles for testing and demonstration."""
    
    def __init__(self, random_seed=42):
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # News categories
        self.categories = [
            'technology', 'sports', 'politics', 'entertainment', 
            'business', 'science', 'health', 'world'
        ]
        
        # Sample headlines and content templates by category
        self.templates = {
            'technology': {
                'headlines': [
                    "New AI breakthrough in {tech_topic}",
                    "Tech giant announces {tech_product}",
                    "Startup raises ${amount}M for {tech_topic}",
                    "Revolutionary {tech_product} changes the game",
                    "Cyber attack hits major {tech_company}",
                    "Scientists develop new {tech_topic} technology",
                    "Social media platform updates {feature}",
                    "Electric vehicle sales reach record high"
                ],
                'content_starters': [
                    "Researchers at a leading university have developed",
                    "A new study reveals that artificial intelligence can",
                    "The technology industry is witnessing unprecedented growth",
                    "Cybersecurity experts are warning about",
                    "Innovation in the field of"
                ]
            },
            'sports': {
                'headlines': [
                    "{team} defeats {opponent} in thrilling match",
                    "Championship final set for {date}",
                    "Star player {name} announces retirement",
                    "Record broken in {sport} competition",
                    "New stadium opens with spectacular ceremony",
                    "Trade deal sends {name} to {team}",
                    "Olympic preparations underway in {location}",
                    "Injury sidelines top athlete for season"
                ],
                'content_starters': [
                    "In an exciting display of athleticism,",
                    "Fans were on the edge of their seats as",
                    "The sporting world was shocked today when",
                    "A new record was set yesterday in",
                    "The competition was fierce as"
                ]
            },
            'politics': {
                'headlines': [
                    "Government announces new {policy} initiative",
                    "Election results show surprising {outcome}",
                    "Political leader addresses {issue}",
                    "New legislation passed on {topic}",
                    "International summit discusses {global_issue}",
                    "Campaign controversy surrounds {politician}",
                    "Voting rights bill advances in legislature",
                    "Foreign policy shift announced"
                ],
                'content_starters': [
                    "In a significant political development,",
                    "Government officials announced today that",
                    "The political landscape changed dramatically when",
                    "Citizens are closely watching as",
                    "International relations experts suggest that"
                ]
            },
            'business': {
                'headlines': [
                    "Stock market reaches new {milestone}",
                    "Company reports {metric} earnings",
                    "Merger between {company1} and {company2}",
                    "Cryptocurrency {action} affects markets",
                    "New trade agreement signed with {country}",
                    "Startup disrupts traditional {industry}",
                    "Economic indicators show {trend}",
                    "Industry leader steps down as CEO"
                ],
                'content_starters': [
                    "Financial markets reacted positively to news that",
                    "Economic analysts are predicting",
                    "The business world was surprised when",
                    "Investors are closely monitoring",
                    "Market volatility increased after"
                ]
            }
        }
        
        # Fill remaining categories with generic templates
        generic_headlines = [
            "Breaking news in {topic}",
            "Experts discuss {subject}",
            "New research reveals {finding}",
            "Community responds to {event}",
            "Analysis: {topic} trends"
        ]
        
        generic_starters = [
            "Recent developments indicate that",
            "A comprehensive analysis shows",
            "Experts in the field believe that",
            "New information suggests that",
            "The latest findings reveal"
        ]
        
        for category in self.categories:
            if category not in self.templates:
                self.templates[category] = {
                    'headlines': generic_headlines,
                    'content_starters': generic_starters
                }
    
    def generate_article_content(self, category, length='medium'):
        """Generate article content based on category and desired length."""
        content_parts = []
        
        # Start with template starter
        starter = random.choice(self.templates[category]['content_starters'])
        content_parts.append(starter)
        
        # Add category-specific content
        if category == 'technology':
            topics = ["artificial intelligence", "machine learning", "blockchain", 
                     "quantum computing", "cybersecurity", "cloud computing", 
                     "Internet of Things", "virtual reality"]
            content_parts.append(f" {random.choice(topics)} technology is advancing rapidly.")
            content_parts.append(" Industry experts believe this development could revolutionize how we work and live.")
            
        elif category == 'sports':
            sports_terms = ["championship", "tournament", "league", "competition", 
                           "playoffs", "season", "training", "performance"]
            content_parts.append(f" the {random.choice(sports_terms)} has captured global attention.")
            content_parts.append(" Athletes and fans alike are excited about the upcoming events.")
            
        elif category == 'politics':
            political_topics = ["healthcare reform", "economic policy", "climate change", 
                               "education funding", "infrastructure", "social programs"]
            content_parts.append(f" {random.choice(political_topics)} remains a key priority.")
            content_parts.append(" Policymakers are working to address public concerns.")
            
        elif category == 'business':
            business_terms = ["market growth", "innovation", "investment", "strategy", 
                             "expansion", "efficiency", "competition", "sustainability"]
            content_parts.append(f" {random.choice(business_terms)} continues to drive the industry forward.")
            content_parts.append(" Companies are adapting to changing market conditions.")
        
        # Add more content based on length
        if length == 'long':
            content_parts.append(" Furthermore, recent studies have shown significant impact across multiple sectors.")
            content_parts.append(" Stakeholders are carefully monitoring the situation as it develops.")
            content_parts.append(" The long-term implications of these changes are still being evaluated.")
        
        return ' '.join(content_parts)
    
    def generate_headline(self, category):
        """Generate a headline for the given category."""
        template = random.choice(self.templates[category]['headlines'])
        
        # Fill in template placeholders
        placeholders = {
            'tech_topic': random.choice(['machine learning', 'blockchain', 'AI', 'cybersecurity']),
            'tech_product': random.choice(['smartphone', 'software', 'platform', 'device']),
            'tech_company': random.choice(['corporation', 'startup', 'platform']),
            'feature': random.choice(['privacy settings', 'user interface', 'security features']),
            'amount': random.choice(['10', '25', '50', '100']),
            'team': random.choice(['Hawks', 'Lions', 'Eagles', 'Tigers']),
            'opponent': random.choice(['Wolves', 'Bears', 'Giants', 'Panthers']),
            'name': random.choice(['Johnson', 'Smith', 'Williams', 'Brown']),
            'sport': random.choice(['basketball', 'football', 'tennis', 'soccer']),
            'location': random.choice(['New York', 'Los Angeles', 'Chicago', 'Miami']),
            'policy': random.choice(['healthcare', 'education', 'environmental', 'economic']),
            'outcome': random.choice(['victory', 'upset', 'development', 'change']),
            'issue': random.choice(['climate change', 'healthcare', 'economy', 'education']),
            'topic': random.choice(['healthcare', 'environment', 'education', 'infrastructure']),
            'politician': random.choice(['Senator', 'Governor', 'Mayor', 'Representative']),
            'milestone': random.choice(['high', 'record', 'milestone', 'achievement']),
            'metric': random.choice(['strong', 'record', 'impressive', 'surprising']),
            'company1': random.choice(['TechCorp', 'DataSystems', 'CloudTech', 'AIInnovate']),
            'company2': random.choice(['Solutions Inc', 'Digital Dynamics', 'Smart Systems', 'Future Tech']),
            'action': random.choice(['surge', 'drop', 'volatility', 'adoption']),
            'country': random.choice(['Japan', 'Germany', 'Canada', 'Australia']),
            'industry': random.choice(['retail', 'manufacturing', 'finance', 'healthcare']),
            'trend': random.choice(['growth', 'stability', 'recovery', 'expansion']),
            'subject': random.choice(['innovation', 'sustainability', 'efficiency', 'growth']),
            'finding': random.choice(['breakthrough', 'insight', 'discovery', 'trend']),
            'event': random.choice(['announcement', 'development', 'initiative', 'program']),
            'date': random.choice(['next month', 'this weekend', 'next week', 'tomorrow'])
        }
        
        for key, value in placeholders.items():
            template = template.replace('{' + key + '}', value)
        
        return template
    
    def generate_news_dataset(self, n_articles=500, save_path=None):
        """Generate a complete news dataset."""
        articles = []
        
        for i in range(n_articles):
            category = random.choice(self.categories)
            
            article = {
                'id': f"article_{i+1:04d}",
                'title': self.generate_headline(category),
                'content': self.generate_article_content(category),
                'category': category,
                'author': f"{random.choice(['John', 'Jane', 'Alex', 'Sarah'])} {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Davis'])}",
                'published_date': datetime.now() - timedelta(days=random.randint(1, 365)),
                'views': random.randint(100, 10000),
                'likes': random.randint(10, 500)
            }
            
            articles.append(article)
        
        df = pd.DataFrame(articles)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Dataset saved to {save_path}")
        
        return df

def generate_user_interactions(articles_df, n_users=100, n_interactions=1000):
    """Generate synthetic user interaction data."""
    np.random.seed(42)
    
    interactions = []
    user_ids = [f"user_{i:04d}" for i in range(1, n_users + 1)]
    
    # Create user preferences (some users prefer certain categories)
    user_preferences = {}
    for user_id in user_ids:
        # 70% chance of having preferred categories
        if random.random() < 0.7:
            n_preferred = random.randint(1, 3)
            preferred = random.sample(articles_df['category'].unique().tolist(), n_preferred)
            user_preferences[user_id] = preferred
        else:
            user_preferences[user_id] = []
    
    for _ in range(n_interactions):
        user_id = random.choice(user_ids)
        
        # Select article based on user preferences
        user_prefs = user_preferences[user_id]
        if user_prefs and random.random() < 0.8:  # 80% chance to pick from preferred categories
            category_articles = articles_df[articles_df['category'].isin(user_prefs)]
            if len(category_articles) > 0:
                article = category_articles.sample(1).iloc[0]
            else:
                article = articles_df.sample(1).iloc[0]
        else:
            article = articles_df.sample(1).iloc[0]
        
        # Generate rating based on preference match
        base_rating = 3.0
        if article['category'] in user_preferences[user_id]:
            base_rating += random.uniform(0.5, 1.5)  # Higher rating for preferred categories
        else:
            base_rating += random.uniform(-1.0, 1.0)
        
        # Add some noise
        rating = base_rating + random.normal(0, 0.5)
        rating = max(1, min(5, int(round(rating))))
        
        interactions.append({
            'user_id': user_id,
            'article_id': article['id'],
            'rating': rating,
            'timestamp': datetime.now() - timedelta(days=random.randint(1, 180)),
            'category': article['category']
        })
    
    return pd.DataFrame(interactions)

def create_sample_data(data_dir="data"):
    """Create all sample datasets for the project."""
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating sample news dataset...")
    generator = NewsDataGenerator()
    
    # Generate articles dataset
    articles_df = generator.generate_news_dataset(
        n_articles=500,
        save_path=os.path.join(data_dir, "news_articles.csv")
    )
    
    # Generate user interactions
    print("Generating user interactions...")
    interactions_df = generate_user_interactions(
        articles_df,
        n_users=150,
        n_interactions=2000
    )
    interactions_df.to_csv(os.path.join(data_dir, "user_interactions.csv"), index=False)
    print(f"User interactions saved to {os.path.join(data_dir, 'user_interactions.csv')}")
    
    # Generate test data
    print("Generating test dataset...")
    test_df = generator.generate_news_dataset(
        n_articles=100,
        save_path=os.path.join(data_dir, "test_articles.csv")
    )
    
    # Create a small labeled dataset for quick testing
    sample_df = articles_df.sample(n=50).copy()
    sample_df.to_csv(os.path.join(data_dir, "sample_articles.csv"), index=False)
    
    print("\nDataset Summary:")
    print(f"- Main articles dataset: {len(articles_df)} articles")
    print(f"- User interactions: {len(interactions_df)} interactions")
    print(f"- Test articles: {len(test_df)} articles")
    print(f"- Sample articles: {len(sample_df)} articles")
    
    print("\nCategory distribution:")
    print(articles_df['category'].value_counts())
    
    return articles_df, interactions_df, test_df

if __name__ == "__main__":
    create_sample_data()