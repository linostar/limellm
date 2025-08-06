# Sample Python code for testing

def factorial(n):
    """Calculate factorial using recursion."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


class DataProcessor:
    """A sample data processing class."""
    
    def __init__(self, name="default"):
        self.name = name
        self.processed_count = 0
    
    def process_data(self, data):
        """Process a list of data items."""
        results = []
        for item in data:
            if isinstance(item, str):
                results.append(item.upper())
            elif isinstance(item, (int, float)):
                results.append(item * 2)
            else:
                results.append(str(item))
        
        self.processed_count += len(data)
        return results
    
    def get_stats(self):
        """Get processing statistics."""
        return {
            "name": self.name,
            "processed_count": self.processed_count
        }


# Example usage
if __name__ == "__main__":
    processor = DataProcessor("test_processor")
    
    # Test with different data types
    test_data = ["hello", 42, 3.14, True]
    results = processor.process_data(test_data)
    
    print("Results:", results)
    print("Stats:", processor.get_stats())
    
    # Test factorial
    for i in range(6):
        print(f"{i}! = {factorial(i)}")