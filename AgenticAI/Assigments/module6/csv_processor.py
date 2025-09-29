#!/usr/bin/env python3
"""
CSV Processor for Data Analysis and LLM Applications

This module provides comprehensive CSV processing capabilities including
data cleaning, analysis, transformation, and preparation for LLM training/inference.
Essential for handling structured data in AI applications.
"""

import csv
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import warnings


@dataclass
class ColumnProfile:
    """Profile information for a CSV column."""
    name: str
    data_type: str
    unique_count: int
    null_count: int
    sample_values: List[Any]
    statistics: Dict[str, Any] = field(default_factory=dict)
    quality_issues: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment."""
    total_rows: int
    total_columns: int
    column_profiles: List[ColumnProfile]
    overall_quality_score: float
    recommendations: List[str]
    duplicate_rows: int
    missing_data_percentage: float


class CSVProcessor:
    """
    Comprehensive CSV processor for data analysis and LLM applications.
    
    This processor demonstrates key concepts:
    - Data quality assessment and cleaning
    - Statistical analysis and profiling
    - Data transformation for ML/LLM use
    - Text preprocessing for NLP tasks
    - Feature engineering and extraction
    """
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.file_path: Optional[Path] = None
        self.processing_log: List[Dict] = []
        
    def load_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load CSV file with intelligent parameter detection.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional pandas read_csv parameters
            
        Returns:
            Loaded DataFrame
        """
        self.file_path = Path(file_path)
        
        # Default parameters for robust CSV reading
        default_params = {
            'encoding': 'utf-8',
            'low_memory': False,
            'parse_dates': False,  # Disable automatic date parsing to avoid warnings
            'date_format': None,  # Disable date format inference
            'na_values': ['', 'NULL', 'null', 'N/A', 'n/a', 'NA', 'nan', 'NaN'],
            'keep_default_na': True,
            'skipinitialspace': True
        }
        
        # Override with user parameters
        params = {**default_params, **kwargs}
        
        try:
            # Try to load with default parameters
            self.df = pd.read_csv(file_path, **params)
            self.original_df = self.df.copy()
            
            self._log_operation("load_csv", f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    params['encoding'] = encoding
                    self.df = pd.read_csv(file_path, **params)
                    self.original_df = self.df.copy()
                    
                    self._log_operation("load_csv", 
                                      f"Loaded with {encoding} encoding: {len(self.df)} rows, {len(self.df.columns)} columns")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode file with any common encoding")
        
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")
        
        return self.df
    
    def _log_operation(self, operation: str, details: str, metadata: Dict = None) -> None:
        """Log processing operations for audit trail."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details,
            'metadata': metadata or {}
        }
        self.processing_log.append(log_entry)
    
    def analyze_data_quality(self) -> DataQualityReport:
        """
        Perform comprehensive data quality analysis.
        
        Returns:
            Detailed data quality report
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        column_profiles = []
        quality_issues = []
        
        for column in self.df.columns:
            profile = self._profile_column(column)
            column_profiles.append(profile)
            
            if profile.quality_issues:
                quality_issues.extend(profile.quality_issues)
        
        # Calculate overall metrics
        total_cells = len(self.df) * len(self.df.columns)
        missing_cells = self.df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        duplicate_rows = len(self.df) - len(self.df.drop_duplicates())
        
        # Calculate quality score (0-100)
        quality_score = self._calculate_quality_score(column_profiles, missing_percentage, duplicate_rows)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(column_profiles, missing_percentage, duplicate_rows)
        
        report = DataQualityReport(
            total_rows=len(self.df),
            total_columns=len(self.df.columns),
            column_profiles=column_profiles,
            overall_quality_score=quality_score,
            recommendations=recommendations,
            duplicate_rows=duplicate_rows,
            missing_data_percentage=missing_percentage
        )
        
        self._log_operation("analyze_data_quality", f"Quality score: {quality_score:.1f}/100")
        
        return report
    
    def _profile_column(self, column: str) -> ColumnProfile:
        """Create detailed profile for a single column."""
        series = self.df[column]
        
        # Basic statistics
        unique_count = series.nunique()
        null_count = series.isnull().sum()
        
        # Infer data type
        data_type = self._infer_data_type(series)
        
        # Sample values (non-null)
        sample_values = series.dropna().head(5).tolist()
        
        # Statistics based on data type
        statistics = {}
        quality_issues = []
        
        if data_type in ['integer', 'float']:
            statistics = {
                'mean': float(series.mean()) if not series.empty else 0,
                'median': float(series.median()) if not series.empty else 0,
                'std': float(series.std()) if not series.empty else 0,
                'min': float(series.min()) if not series.empty else 0,
                'max': float(series.max()) if not series.empty else 0,
                'quartiles': series.quantile([0.25, 0.5, 0.75]).tolist() if not series.empty else []
            }
            
            # Check for outliers (values beyond 3 standard deviations)
            if len(series.dropna()) > 0:
                mean = series.mean()
                std = series.std()
                outliers = series[(np.abs(series - mean) > 3 * std)].count()
                if outliers > 0:
                    quality_issues.append(f"Contains {outliers} potential outliers")
        
        elif data_type == 'string':
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                statistics = {
                    'avg_length': float(non_null_series.str.len().mean()),
                    'min_length': int(non_null_series.str.len().min()),
                    'max_length': int(non_null_series.str.len().max()),
                    'most_common': non_null_series.value_counts().head(3).to_dict()
                }
                
                # Check for inconsistent formatting
                if self._has_inconsistent_formatting(non_null_series):
                    quality_issues.append("Inconsistent text formatting detected")
        
        elif data_type == 'datetime':
            non_null_series = pd.to_datetime(series, errors='coerce')
            if not non_null_series.isna().all():
                statistics = {
                    'earliest': non_null_series.min().isoformat() if pd.notna(non_null_series.min()) else None,
                    'latest': non_null_series.max().isoformat() if pd.notna(non_null_series.max()) else None,
                    'date_range_days': (non_null_series.max() - non_null_series.min()).days if pd.notna(non_null_series.min()) and pd.notna(non_null_series.max()) else 0
                }
        
        # General quality checks
        if null_count > len(series) * 0.5:
            quality_issues.append(f"High missing data rate: {(null_count/len(series)*100):.1f}%")
        
        if unique_count == 1 and len(series) > 1:
            quality_issues.append("Column has only one unique value")
        
        if unique_count == len(series) and data_type == 'string':
            quality_issues.append("All values are unique (possible identifier column)")
        
        return ColumnProfile(
            name=column,
            data_type=data_type,
            unique_count=unique_count,
            null_count=null_count,
            sample_values=sample_values,
            statistics=statistics,
            quality_issues=quality_issues
        )
    
    def _infer_data_type(self, series: pd.Series) -> str:
        """Infer the most appropriate data type for a series."""
        # Remove null values for type inference
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return 'unknown'
        
        # Try numeric types first
        try:
            pd.to_numeric(non_null)
            if all(isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()) 
                   for x in non_null if pd.notna(x)):
                return 'integer'
            else:
                return 'float'
        except (ValueError, TypeError):
            pass
        
        # Try datetime
        try:
            pd.to_datetime(non_null, infer_datetime_format=True)
            return 'datetime'
        except (ValueError, TypeError):
            pass
        
        # Try boolean
        if set(non_null.astype(str).str.lower().unique()) <= {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}:
            return 'boolean'
        
        # Default to string
        return 'string'
    
    def _has_inconsistent_formatting(self, series: pd.Series) -> bool:
        """Check if string series has inconsistent formatting."""
        if len(series) < 2:
            return False
        
        # Check for mixed case patterns
        has_upper = series.str.contains(r'[A-Z]').any()
        has_lower = series.str.contains(r'[a-z]').any()
        
        if has_upper and has_lower:
            # Check if some are all caps and others are mixed/lower
            all_caps_count = series.str.isupper().sum()
            mixed_case_count = len(series) - all_caps_count
            
            if all_caps_count > 0 and mixed_case_count > 0:
                return True
        
        # Check for inconsistent spacing
        has_multiple_spaces = series.str.contains(r'\s{2,}').any()
        has_leading_trailing_spaces = (series.str.startswith(' ') | series.str.endswith(' ')).any()
        
        return has_multiple_spaces or has_leading_trailing_spaces
    
    def _calculate_quality_score(self, profiles: List[ColumnProfile], 
                                missing_percentage: float, duplicate_rows: int) -> float:
        """Calculate overall data quality score (0-100)."""
        score = 100.0
        
        # Penalize missing data
        score -= missing_percentage * 0.5
        
        # Penalize duplicates
        if len(self.df) > 0:
            duplicate_percentage = (duplicate_rows / len(self.df)) * 100
            score -= duplicate_percentage * 0.3
        
        # Penalize quality issues
        total_issues = sum(len(profile.quality_issues) for profile in profiles)
        score -= total_issues * 2
        
        # Bonus for consistent data types
        consistent_columns = sum(1 for profile in profiles if not profile.quality_issues)
        if len(profiles) > 0:
            consistency_bonus = (consistent_columns / len(profiles)) * 10
            score += consistency_bonus
        
        return max(0.0, min(100.0, score))
    
    def _generate_recommendations(self, profiles: List[ColumnProfile], 
                                 missing_percentage: float, duplicate_rows: int) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        if missing_percentage > 10:
            recommendations.append(f"Consider addressing missing data ({missing_percentage:.1f}% of cells are empty)")
        
        if duplicate_rows > 0:
            recommendations.append(f"Remove {duplicate_rows} duplicate rows")
        
        high_missing_columns = [p.name for p in profiles if p.null_count > len(self.df) * 0.3]
        if high_missing_columns:
            recommendations.append(f"Columns with high missing data: {', '.join(high_missing_columns[:3])}")
        
        inconsistent_columns = [p.name for p in profiles if any('formatting' in issue.lower() for issue in p.quality_issues)]
        if inconsistent_columns:
            recommendations.append(f"Clean text formatting in: {', '.join(inconsistent_columns[:3])}")
        
        outlier_columns = [p.name for p in profiles if any('outlier' in issue.lower() for issue in p.quality_issues)]
        if outlier_columns:
            recommendations.append(f"Review outliers in: {', '.join(outlier_columns[:3])}")
        
        return recommendations
    
    def clean_data(self, operations: List[str] = None) -> pd.DataFrame:
        """
        Apply data cleaning operations.
        
        Args:
            operations: List of cleaning operations to apply
                       ['duplicates', 'missing', 'whitespace', 'outliers', 'formatting']
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        if operations is None:
            operations = ['duplicates', 'whitespace', 'formatting']
        
        cleaned_df = self.df.copy()
        
        for operation in operations:
            if operation == 'duplicates':
                before_count = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                removed = before_count - len(cleaned_df)
                if removed > 0:
                    self._log_operation('remove_duplicates', f"Removed {removed} duplicate rows")
            
            elif operation == 'missing':
                # Fill missing values with appropriate defaults
                for column in cleaned_df.columns:
                    if cleaned_df[column].dtype in ['object']:
                        cleaned_df[column] = cleaned_df[column].fillna('Unknown')
                    elif cleaned_df[column].dtype in ['int64', 'float64']:
                        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
                
                self._log_operation('handle_missing', "Filled missing values")
            
            elif operation == 'whitespace':
                # Clean text columns
                text_columns = cleaned_df.select_dtypes(include=['object']).columns
                for column in text_columns:
                    cleaned_df[column] = cleaned_df[column].astype(str).str.strip()
                    cleaned_df[column] = cleaned_df[column].str.replace(r'\s+', ' ', regex=True)
                
                self._log_operation('clean_whitespace', f"Cleaned whitespace in {len(text_columns)} columns")
            
            elif operation == 'formatting':
                # Standardize text formatting
                text_columns = cleaned_df.select_dtypes(include=['object']).columns
                for column in text_columns:
                    # Convert to title case if it looks like names/titles
                    sample = cleaned_df[column].dropna().head(10)
                    if any(re.match(r'^[A-Za-z\s]+$', str(val)) for val in sample):
                        cleaned_df[column] = cleaned_df[column].str.title()
                
                self._log_operation('standardize_formatting', f"Standardized formatting in {len(text_columns)} columns")
            
            elif operation == 'outliers':
                # Remove extreme outliers (beyond 3 standard deviations)
                numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
                for column in numeric_columns:
                    before_count = len(cleaned_df)
                    mean = cleaned_df[column].mean()
                    std = cleaned_df[column].std()
                    cleaned_df = cleaned_df[np.abs(cleaned_df[column] - mean) <= 3 * std]
                    removed = before_count - len(cleaned_df)
                    if removed > 0:
                        self._log_operation('remove_outliers', f"Removed {removed} outliers from {column}")
        
        self.df = cleaned_df
        return self.df
    
    def transform_for_llm(self, text_columns: List[str] = None, 
                         combine_columns: bool = True) -> List[Dict[str, Any]]:
        """
        Transform data for LLM training or inference.
        
        Args:
            text_columns: Specific columns to include as text
            combine_columns: Whether to combine all text into single field
            
        Returns:
            List of dictionaries suitable for LLM processing
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        if text_columns is None:
            # Auto-detect text columns
            text_columns = list(self.df.select_dtypes(include=['object']).columns)
        
        transformed_data = []
        
        for idx, row in self.df.iterrows():
            record = {'row_id': idx}
            
            if combine_columns:
                # Combine all text columns into a single text field
                text_parts = []
                for col in text_columns:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        text_parts.append(f"{col}: {row[col]}")
                
                record['text'] = ' | '.join(text_parts)
                
                # Add numeric fields separately
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if pd.notna(row[col]):
                        record[col] = row[col]
            else:
                # Keep columns separate
                for col in self.df.columns:
                    if pd.notna(row[col]):
                        record[col] = row[col]
            
            transformed_data.append(record)
        
        self._log_operation('transform_for_llm', f"Transformed {len(transformed_data)} records")
        
        return transformed_data
    
    def extract_features(self, text_column: str) -> pd.DataFrame:
        """
        Extract features from text column for analysis.
        
        Args:
            text_column: Name of text column to analyze
            
        Returns:
            DataFrame with extracted features
        """
        if self.df is None or text_column not in self.df.columns:
            raise ValueError(f"Column {text_column} not found in data")
        
        text_series = self.df[text_column].dropna().astype(str)
        
        features = pd.DataFrame(index=text_series.index)
        
        # Basic text features
        features['text_length'] = text_series.str.len()
        features['word_count'] = text_series.str.split().str.len()
        features['sentence_count'] = text_series.str.count(r'[.!?]+') + 1
        features['avg_word_length'] = features['text_length'] / features['word_count']
        
        # Advanced features
        features['uppercase_ratio'] = text_series.str.count(r'[A-Z]') / features['text_length']
        features['punctuation_ratio'] = text_series.str.count(r'[^\w\s]') / features['text_length']
        features['digit_ratio'] = text_series.str.count(r'\d') / features['text_length']
        
        # Sentiment indicators (simple approach)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'poor', 'worst']
        
        features['positive_word_count'] = text_series.str.lower().str.count('|'.join(positive_words))
        features['negative_word_count'] = text_series.str.lower().str.count('|'.join(negative_words))
        
        self._log_operation('extract_features', f"Extracted features for {len(features)} records")
        
        return features
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of the dataset."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        quality_report = self.analyze_data_quality()
        
        summary = {
            'file_info': {
                'path': str(self.file_path) if self.file_path else 'Unknown',
                'size_mb': self.file_path.stat().st_size / (1024*1024) if self.file_path and self.file_path.exists() else 0,
                'last_modified': datetime.fromtimestamp(self.file_path.stat().st_mtime).isoformat() if self.file_path and self.file_path.exists() else 'Unknown'
            },
            'data_overview': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024*1024),
                'data_types': self.df.dtypes.value_counts().to_dict()
            },
            'quality_metrics': {
                'overall_score': quality_report.overall_quality_score,
                'missing_data_percentage': quality_report.missing_data_percentage,
                'duplicate_rows': quality_report.duplicate_rows,
                'columns_with_issues': len([p for p in quality_report.column_profiles if p.quality_issues])
            },
            'column_summary': {
                profile.name: {
                    'type': profile.data_type,
                    'unique_values': profile.unique_count,
                    'missing_count': profile.null_count,
                    'sample_values': profile.sample_values[:3]
                }
                for profile in quality_report.column_profiles
            },
            'recommendations': quality_report.recommendations,
            'processing_log': self.processing_log[-5:]  # Last 5 operations
        }
        
        return summary
    
    def export_processed_data(self, output_path: str, format: str = 'csv') -> None:
        """
        Export processed data in various formats.
        
        Args:
            output_path: Path for output file
            format: Output format ('csv', 'json', 'parquet')
        """
        if self.df is None:
            raise ValueError("No data to export")
        
        output_path = Path(output_path)
        
        if format.lower() == 'csv':
            self.df.to_csv(output_path, index=False, encoding='utf-8')
        elif format.lower() == 'json':
            self.df.to_json(output_path, orient='records', indent=2)
        elif format.lower() == 'parquet':
            self.df.to_parquet(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self._log_operation('export_data', f"Exported to {output_path} ({format} format)")
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a CSV file and return summary information.
        Compatible method for support agent integration.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary with processing results and statistics
        """
        try:
            # Load and analyze the CSV file
            df = self.load_csv(file_path)
            quality_report = self.analyze_data_quality()
            
            # Generate summary
            summary = self.generate_summary()
            
            # Return compatible format for support agent
            return {
                'status': 'success',
                'rows_processed': len(df),
                'columns_processed': len(df.columns),
                'data_quality_score': quality_report.overall_quality_score,
                'file_type': 'csv',
                'summary': summary,
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'file_type': 'csv',
                'processing_timestamp': datetime.now().isoformat()
            }


def create_sample_csv():
    """Create a sample CSV file for demonstration."""
    sample_data = {
        'customer_id': [f'CUST_{i:04d}' for i in range(1, 101)],
        'name': [
            'John Smith', 'jane doe', 'BOB JOHNSON', 'Mary  Williams', ' Alice Brown',
            'Charlie Wilson', 'diana davis', 'FRANK MILLER', 'Grace Taylor', 'Henry Lee'
        ] * 10,
        'email': [
            'john@email.com', 'jane@test.com', '', 'mary@company.org', 'alice@mail.net',
            'charlie@work.com', 'diana@email.co', 'frank@business.com', '', 'henry@test.org'
        ] * 10,
        'age': [25, 34, 28, None, 45, 67, 23, 41, 38, 29] * 10,
        'purchase_amount': [
            150.50, 89.99, 1200.00, 45.75, 567.80, 23.40, 890.25, 2500.00, 67.90, 345.60
        ] * 10,
        'feedback': [
            'Great service!', 'Product was okay', 'Terrible experience', 'Amazing quality',
            'Good value for money', 'Poor customer support', 'Excellent product', 
            'Not worth the price', 'Outstanding service', 'Average experience'
        ] * 10,
        'registration_date': [
            '2023-01-15', '2023-02-20', '2023-01-30', '2023-03-10', '2023-02-14',
            '2023-01-25', '2023-03-05', '2023-02-28', '2023-01-20', '2023-03-15'
        ] * 10
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some duplicates and data quality issues
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)  # Add duplicates
    df.loc[10:15, 'email'] = None  # Add missing emails
    df.loc[20, 'age'] = 150  # Add outlier
    
    df.to_csv('sample_customer_data.csv', index=False)
    return 'sample_customer_data.csv'


def demo_csv_processing():
    """Demonstrate CSV processing capabilities."""
    print("CSV Processor Demo")
    print("=" * 50)
    
    # Create sample data
    sample_file = create_sample_csv()
    print(f"Created sample file: {sample_file}")
    print()
    
    # Initialize processor
    processor = CSVProcessor()
    
    # Load data
    print("Loading CSV data...")
    df = processor.load_csv(sample_file)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print()
    
    # Analyze data quality
    print("Analyzing data quality...")
    quality_report = processor.analyze_data_quality()
    print(f"Overall Quality Score: {quality_report.overall_quality_score:.1f}/100")
    print(f"Missing Data: {quality_report.missing_data_percentage:.1f}%")
    print(f"Duplicate Rows: {quality_report.duplicate_rows}")
    print()
    
    print("Column Profiles:")
    for profile in quality_report.column_profiles[:3]:  # Show first 3 columns
        print(f"  {profile.name} ({profile.data_type}):")
        print(f"    - Unique values: {profile.unique_count}")
        print(f"    - Missing values: {profile.null_count}")
        if profile.quality_issues:
            print(f"    - Issues: {', '.join(profile.quality_issues)}")
        print(f"    - Sample values: {profile.sample_values}")
        print()
    
    # Clean data
    print("Cleaning data...")
    cleaned_df = processor.clean_data(['duplicates', 'whitespace', 'formatting'])
    print(f"After cleaning: {len(cleaned_df)} rows")
    print()
    
    # Extract text features
    print("Extracting text features from feedback column...")
    features = processor.extract_features('feedback')
    print("Text Features (first 5 rows):")
    print(features.head())
    print()
    
    # Transform for LLM
    print("Transforming data for LLM use...")
    llm_data = processor.transform_for_llm(['name', 'feedback'])
    print("Sample LLM-ready records:")
    for i, record in enumerate(llm_data[:3]):
        print(f"  Record {i+1}: {record}")
    print()
    
    # Generate summary
    print("Generating comprehensive summary...")
    summary = processor.generate_summary()
    print("Dataset Summary:")
    print(f"  - File size: {summary['file_info']['size_mb']:.2f} MB")
    print(f"  - Memory usage: {summary['data_overview']['memory_usage_mb']:.2f} MB")
    print(f"  - Quality score: {summary['quality_metrics']['overall_score']:.1f}/100")
    print(f"  - Data types: {summary['data_overview']['data_types']}")
    print()
    
    print("Recommendations:")
    for rec in summary['recommendations']:
        print(f"  - {rec}")
    print()
    
    # Export processed data
    processor.export_processed_data('processed_data.csv', 'csv')
    processor.export_processed_data('processed_data.json', 'json')
    print("Exported processed data to CSV and JSON formats")
    
    # Export summary
    with open('data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("Exported data summary to data_summary.json")


if __name__ == "__main__":
    demo_csv_processing()