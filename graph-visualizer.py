from Tkinter import *
import tkMessageBox
import math
from tkFileDialog import askopenfilename, asksaveasfilename

 #/**
 # GLOBALS
 #/
state = "node"
link_start = (0, 0)
clicked = None
label = 'A'
node_size = 40
node_map = {}
temp = None
all_labels = []
buttonRow = 1
buttonCol = 1
all_ovals = []
all_lines = []
srch = None
step_count = 0


#/**
# Searches
#/
class Astar():

    def __init__(self, frm, to, tree):
        if frm == "" or to == "":
            tkMessageBox.showinfo(title="Warning", message="Please select a start and finish node for A*")
            self.initialized = False
            return

        self.tree = tree
        self.openset = set([frm])
        self.closedset = set()
        self.g_scores = {}
        self.f_scores = {}
        self.came_from = {}
        self.came_from[frm] = None
        self.g_scores[frm] = 0
        self.f_scores[frm] = self.get_dist_between(frm, to)
        self.dest = to
        self.initialized = True
        self.complete = False

    def node(self, name):
        if name in self.tree:
            return self.tree[name]['inst']
        else:
            return None

    def get_dist_between(self, a, b):
        node1 = self.tree[a]['inst']
        node2 = self.tree[b]['inst']
        dist = math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
        return dist

    def step(self):
        if not self.initialized:
            return False
        if self.complete:
            return True

        def sort_cmp(x, y):
            return x if self.f_scores[x] < self.f_scores[y] else y

        if self.openset:
            current = reduce(sort_cmp, self.openset)
            if current == self.dest:
                self.complete = True
                return self.get_current_path(current)
            self.openset.remove(current)
            self.closedset.add(current)
            current_node = self.tree[current]['inst']
            for neighbor in current_node.neighbors:
                if neighbor.label in self.closedset:
                    continue
                tentative_g_score = self.g_scores[current] + self.get_dist_between(current, neighbor.label)
                if not neighbor.label in self.openset or tentative_g_score < self.g_scores[neighbor.label]:
                    self.g_scores[neighbor.label] = tentative_g_score
                    self.f_scores[neighbor.label] = self.g_scores[neighbor.label] + self.get_dist_between(neighbor.label, self.dest)
                    self.openset.add(neighbor.label)
                    self.came_from[neighbor.label] = current

            return self.get_current_path(current)
        return False

    def get_current_path(self, label):
        reset()
        ovals = []
        lines = []
        current = self.node(label)
        came_from = self.node(self.came_from[label])
        if came_from is None:
            return None
        while(came_from != None):
            ovals.extend([current.oval, came_from.oval])
            lines.append(current.neighbors[came_from])
            current = came_from
            came_from = self.node(self.came_from[current.label])

        return ovals, lines


class Prim():

    def __init__(self, frm, to, tree):
        import operator
        if frm == "":
            self.edges = []
            tkMessageBox.showinfo(title="Warning", message="Please select an intial node for Prims Algorithm")
            return
        for item in tree:
            edges = {}
            for label, node in tree.items():
                for neighbor, dist in node['neighbors'].items():
                    if not (label, neighbor) in edges and not (neighbor, label) in edges:
                        edges[label, neighbor] = dist

        self.edges = list(sorted(edges.iteritems(), key=operator.itemgetter(1)))
        self.tree = tree
        self.set = set(frm)
        self.selected = []
        self.last = None

    def valid_edges(self, edge):
        nodes, dist = edge
        e1, e2 = nodes
        if (e1 in self.set and e2 in self.set):
            return False
        if (e1 not in self.set and e2 not in self.set):
            return False
        return True

    def step(self):
        edges = filter(self.valid_edges, self.edges)
        if len(edges) > 0:
            edge, dist = edges.pop(0)
            v1, v2 = edge
            self.set.add(v1)
            self.set.add(v2)
            node1 = self.tree[v1]['inst']
            node2 = self.tree[v2]['inst']
            line = node1.neighbors[node2]

            return ((node1.oval, node2.oval), (line,))
        return True


class Kruskal():
    def __init__(self, frm, to, tree):
        import operator
        for item in tree:
            edges = {}
            for label, node in tree.items():
                for neighbor, dist in node['neighbors'].items():
                    if not (label, neighbor) in edges and not (neighbor, label) in edges:
                        edges[label, neighbor] = dist

        self.tree = tree
        self.edges = list(reversed(sorted(edges.iteritems(), key=operator.itemgetter(1))))
        self.forest = []

    def step(self):
        if len(self.edges) > 0:
            edge, dist = self.edges.pop()
            v1, v2 = edge
            tree1 = tree2 = None
            for tree in self.forest:
                if v1 in tree:
                    if v2 in tree:
                        return self.step()
                    tree1 = tree
                if v2 in tree:
                    tree2 = tree
            new_tree = set([v1, v2])
            if tree1 is not None:
                self.forest.remove(tree1)
                new_tree.update(tree1)
            if tree2 is not None:
                self.forest.remove(tree2)
                new_tree.update(tree2)
            self.forest.append(new_tree)
            node1 = self.tree[v1]['inst']
            node2 = self.tree[v2]['inst']
            line = node1.neighbors[node2]

            return ((node1.oval, node2.oval), (line,))
        print self.forest
        return True


class Dijkstra():
    def __init__(self, frm, to, tree):
        if frm == "" or to == "":
            tkMessageBox.showinfo(title="Warning", message="Please select a start and finish node for A*")
            self.initialized = False
            return

        self.tree = tree
        self.dest = to
        self.initialized = True
        self.complete = False
        self.all_labels = [node for node in tree]
        self.dist = {}
        self.prev = {}
        for label in self.all_labels:
            self.dist[label] = float('inf')
            self.prev[label] = None
        self.dist[frm] = 0

    def get_dist_between(self, a, b):
        node1 = self.tree[a]['inst']
        node2 = self.tree[b]['inst']
        dist = math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
        return dist

    def node(self, name):
        if name in self.tree:
            return self.tree[name]['inst']
        else:
            return None

    def get_path_from(self, label):
        reset()
        ovals = []
        lines = []
        current = self.node(label)
        previous = self.node(self.prev[label])
        if previous is None:
            return None
        while(previous != None):
            ovals.extend([current.oval, previous.oval])
            lines.append(current.neighbors[previous])
            current = previous
            previous = self.node(self.prev[current.label])
        return ovals, lines

    def step(self):

        if self.complete:
            return True
        if self.all_labels:
            def get_smallest(x, y):
                return x if self.dist[x] < self.dist[y] else y
            smallest = reduce(get_smallest, self.all_labels)
            if(smallest == self.dest):
                self.complete = True
            if(self.dist[smallest] == float('inf')):
                return False
            self.all_labels.remove(smallest)
            for neighbor_node in self.tree[smallest]['inst'].neighbors:
                neighbor = neighbor_node.label
                alt_dist = self.dist[smallest] + self.get_dist_between(smallest, neighbor)
                if alt_dist < self.dist[neighbor]:
                    self.dist[neighbor] = alt_dist
                    self.prev[neighbor] = smallest
            return self.get_path_from(smallest)
        return True


class Node():

    def __init__(self, x, y, lbel=None):
        global all_ovals
        self.x = x
        self.y = y
        self.update_position
        self.oval = canvas.create_oval(x, y, x + node_size, y + node_size, fill="green")
        all_ovals.append(self.oval)
        self.label = label if lbel == None else lbel
        self.text = canvas.create_text(x + node_size / 2, y - node_size / 2, text=self.label)
        self.neighbors = {}

        all_labels.append(self.label)
        ff["menu"].add_command(label=label, command=lambda temp=label: ff.setvar(ff.cget("textvariable"), value=temp))
        tt["menu"].add_command(label=label, command=lambda temp=label: tt.setvar(tt.cget("textvariable"), value=temp))
        increment_label()
        self.update_position()

    def update_position(self):
        canvas.coords(self.oval, self.x, self.y, self.x + node_size, self.y + node_size)
        canvas.coords(self.text, self.x + node_size / 2, self.y - node_size / 2)

        for neighbor, line in self.neighbors.items():
            x1, y1 = self.center()
            x2, y2 = neighbor.center()
            canvas.coords(line, x1, y1, x2, y2)

    def create_line_to(self, node2):
        global all_lines
        if(node2 in self.neighbors):
            return
        self.neighbors[node2] = canvas.create_line(self.center(), node2.center())
        node2.neighbors[self] = self.neighbors[node2]
        all_lines.append(node2.neighbors[self])

    def center(self):
        return self.x + node_size / 2, self.y + node_size / 2

    def minneighbor(self):
        neighbors = [neighbor.depth for neighbor in self.neighbors if neighbor != self.previous]
        return min(neighbors) if len(neighbors) else float('inf')

    def minlowpoint(self):
        neighbors = [neighbor.lowest for neighbor in self.neighbors if neighbor != self.previous]
        return min(neighbors) if len(neighbors) else float('inf')


def reset():
    for oval in all_ovals:
        canvas.itemconfig(oval, fill="green")
    for line in all_lines:
        canvas.itemconfig(line, fill="black")


def set_state_node(event):
    global state
    state = "node"


def set_state_remove_node(event):
    global state
    state = "removenode"


def set_state_remove_link(event):
    global state
    state = "removelink"


def search(event):
    global srch, step_count
    reset()
    step_count = 0
    search_cls = searches[variable.get()]
    frm = ff.cget("text")
    to = tt.cget("text")
    srch = search_cls(frm, to, get_map())
    root.after(5, search_step)


def search_step():
    global step_count
    res = srch.step()
    if res is False:
        tkMessageBox.showinfo(title="Warning", message="Algorithm failed!")
        return
    if res is True:
        tkMessageBox.showinfo(title="Success", message="Found solution in %s steps " % (str(step_count)))
        return
    if res is None:
        root.after(1, search_step)
        return
    ovals, lines = res
    for item in ovals:
        canvas.itemconfig(item, fill="red")
    for line in lines:
        canvas.itemconfig(line, fill="green")
    step_count += 1
    root.after(speed.get(), search_step)


def set_state_links(event):
    global state
    state = "link"


def set_state_move(event):
    global state
    state = "move"


def add_button(name, handler):
    global buttonRow
    but = Button(root, text=name)
    but.bind("<1>", handler)
    but.grid(row=buttonRow, column=buttonCol)
    buttonRow += 1


def clickedOval(event):
    global clicked
    clicked = canvas.find_withtag(CURRENT)[0]


def increment_label():
    global label
    label = chr(ord(label) + 1)


def add_node(x, y, label=None):
    node = Node(x, y, lbel=label)
    node_map[node.oval] = node
    canvas.tag_bind(node.oval, '<ButtonPress-1>', clickedOval)
    if ff.cget("text") == "":
        frm.set(node.label)
    elif tt.cget("text") == "":
        to.set(node.label)
    return node.oval


def get_line_length(item):
    import math
    item = canvas.coords(item)
    length = math.sqrt((item[0] - item[2]) ** 2 + (item[1] - item[3]) ** 2)
    return round(length, 1)


def get_map():
    nodes = {}

    for nodeid, node in node_map.items():
        node_dct = {}
        node_dct["position"] = node.x, node.y
        node_dct["inst"] = node
        node_dct["neighbors"] = {}
        for neighbor, line in node.neighbors.items():
            node_dct["neighbors"][neighbor.label] = get_line_length(line)
        nodes[node.label] = node_dct

    return nodes


def bind_mouse_events():

    def down(event):
        global link_start, temp
        if(state == "node"):
            add_node(event.x, event.y)
        elif(state == "link"):
            temp = canvas.create_line(event.x, event.y, event.x + 1, event.y + 1)
            link_start = find_node_at(event.x, event.y)

    def find_node_at(x, y):
        overlapping = canvas.find_overlapping(x - node_size, y - node_size, x + node_size, y + node_size)
        for posId in overlapping:
            if posId in node_map:
                return node_map[posId]
        return None

    def drag(event):
        global clicked, temp
        if(clicked is not None and state == "move"):
            node = node_map[clicked]
            node.x = event.x
            node.y = event.y
            node.update_position()
        elif link_start is not None and state == "link":
            canvas.coords(temp, link_start.x + node_size / 2, link_start.y + node_size / 2, event.x, event.y)

    def up(event):
        global clicked
        clicked = None

        try:
            canvas.delete(temp)
        except:
            pass
        if(state == "link"):
            if link_start is not None:
                link_end = find_node_at(event.x, event.y)
                if(link_start is not None and link_end is not None):
                    link_start.create_line_to(link_end)
        elif(state == "removenode"):
            node = find_node_at(event.x, event.y)
            del node_map[node.oval]
            canvas.delete(node.oval)
            canvas.delete(node.text)
            for neighbor, line in node.neighbors.items():
                del neighbor.neighbors[node]
                canvas.delete(line)

    canvas.bind("<Button-1>", down)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", up)


def reconstruct(data):
    global node_map, all_ovals, all_lines, all_labels
    node_map = {}
    all_ovals = []
    all_labels = []
    all_lines = []
    canvas.delete(ALL)
    all_nodes = {}
    for label, node in data.items():
        all_nodes[label] = add_node(node['position'][0], node['position'][1], label=label)

    for label, node in data.items():
        for neighbor in node['neighbors']:
            node1 = node_map[all_nodes[label]]
            node2 = node_map[all_nodes[neighbor]]
            node1.create_line_to(node2)


def save_canvas(event):
    import pickle
    f = open(asksaveasfilename(), 'w')
    pickle.dump(get_map(), f)


def load_canvas(event):
    import pickle
    f = open(askopenfilename(), 'r')
    reconstruct(pickle.load(f))


#/**
 # GUI
 #/
root = Tk()

canvas = Canvas(root, width=900, height=500)
canvas.grid(row=0, column=0)

variable = StringVar(root)
variable.set("A*")

searches = {"A*": Astar, "kruskal": Kruskal, "dijkstra": Dijkstra, "prims": Prim}
w = OptionMenu(root, variable, "A*", "kruskal", "dijkstra", "prims")
w.grid(row=2, column=0)

speed = Scale(root, from_=100, to=2000, orient=HORIZONTAL)
speed.grid(row=3, column=0)

w = Label(root, text="From")
w.grid(row=3, column=2)

w = Label(root, text="Destination")
w.grid(row=3, column=3)

frm = StringVar(root)
frm.set("") # default value

ff = OptionMenu(root, frm, all_labels)
ff.grid(row=4, column=2)

to = StringVar(root)
to.set("") # default value

tt = OptionMenu(root, to, all_labels)
tt.grid(row=4, column=3)


add_button("Add Nodes", set_state_node)
add_button("Add Links", set_state_links)
add_button("Move Nodes", set_state_move)
add_button("Search", search)
buttonCol += 1
buttonRow = 1
add_button("Remove Nodes", set_state_remove_node)
add_button("Save Canvas", save_canvas)
buttonCol += 1
buttonRow -= 1
add_button("Load Canvas", load_canvas)
bind_mouse_events()
root.mainloop()
