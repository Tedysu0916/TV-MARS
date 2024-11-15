class CounterBase():
    def __init__(self,
                 items,
                 items_key:list=None,
                 pos:int=0,
                 **kwargs) -> None:
        self.size = len(items)
        self.items = items
        self.pos = pos
        self.items_key = items_key

    def current(self):
        pos = self.get_pos()
        item = self.items[pos]
        return item

    def get_pos(self, pos:int=None):
        if pos == None:
            pos = self.pos
        if self.items_key != None:
            try:
                pos = self.items_key[pos]
            except:
                pos = None
        return pos

    def update(self, 
               items=None,
               items_key:list=None,
               pos:int=-1):
        if self.items != None:
            self.items = items
            self.size = len(items)
        if pos != -1:
            self.pos = pos
        if items_key != None:
            self.items_key = items_key

    # def next(self, step:int):
    #     if step == 1:
    #         pos = self.get_pos()
    #         res = self.items[pos]
    #         if self.pos < self.size:
    #             self.pos += 1
    #     else:
    #         res = []
    #         extra = 0
    #         if step + self.pos >= self.size:
    #             extra = step + self.pos - self.size
    #         for i in range(step):
    #             pos = self.get_pos()
    #             if self.pos >= self.size:
    #                 res.append(self.empty_item)
    #             else:
    #                 res.append(self.items[pos])
    #                 self.pos += 1
    #         for i in range(extra):
    #             pos = self.get_pos(-self.pos + i)
    #             res.append(self.items[pos])
    #     return res

    # def previous(self, step:int):
    #     if step == 1:
    #         pos = self.get_pos()
    #         res = self.items[pos]
    #         if self.pos >= 0:
    #             self.pos -= 1
    #     else:
    #         res = []
    #         extra = 0
    #         if self.pos - step < 0:
    #             extra = step - self.pos
    #         for i in range(step):
    #             pos = self.get_pos()
    #             if self.pos < 0:
    #                 res.append(self.empty_item)
    #             else:
    #                 res.append(self.items[pos])
    #                 self.pos -= 1
    #         for i in range(step-extra, step):
    #             pos = self.get_pos(i)
    #             res.append(self.items[pos])
    #     return res

            
class CounterLoop(CounterBase):
    def __init__(self,
                 items,
                 items_key:list=None,
                 pos:int=0,
                 **kwargs) -> None:
        super().__init__(items, items_key, pos,**kwargs)

    def next(self, step:int):
        if self.size == 0:
            return {} if abs(step) == 1 else [{}] * 5
        res = []
        isNext = 1 if step >= 0 else -1
        step = abs(step)
        for i in range(step):
            if isNext == -1:
                self.pos += isNext
            pos = self.get_pos()
            res.append(self.items[pos])
            if isNext == 1:
                self.pos += isNext
            if self.pos >= self.size:
                self.pos = 0
            elif self.pos < 0:
                self.pos = self.size - 1
        if step == 1:
            res = res[0]
        return res

    def previous(self, step:int):
        res = self.next(-step)
        return res