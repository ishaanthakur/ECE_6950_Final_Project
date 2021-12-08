class ComparableDict(dict):
    def __eq__(self, other):
        return self.get('id') == other.get('id')
    
    def __lt__(self, other):
        try:
            return self.get('id') < other.get('id')
        except:
            return False