from manim import *
import random
import numpy as np

def get_som_input(N):
    rows = N**3
    X = np.zeros((rows, 3))
    intensities = np.linspace(0,255,N)
    for r in range(N):
        for g in range(N):
            for b in range(N):
                X[(N**2)*r+N*g+b,:] = [intensities[r], intensities[g], intensities[b]]
    filtered = []
    for i in range(N**3):
        if (X[i,0] != X[i,1] or X[i,0] != X[i,2] or X[i,1] != X[i,2]):
            filtered.append(X[i])
    filtered = np.asarray(filtered)
    return filtered

class SomIntro(ThreeDScene):
    def construct(self):
        text = Text("Self-Organizing Map")
        self.play(Write(text))
        self.wait(0.5)
        text2 = MarkupText(
            f'<span underline="single">S</span>elf-<span underline="single">O</span>rganizing <span underline="single">M</span>ap'
        )
        self.add(text2)
        self.remove(text)
        self.wait(1)
        text3 = Text("SOM")
        self.play(Transform(text2, text3))
        #self.wait(1)
        self.play(FadeOut(text2))

        som_img = ImageMobject("colors_20_200_3.png").scale(1.5)
        self.play(FadeIn(som_img))
        self.wait(10)
        self.play(FadeOut(som_img))
        self.wait(2)

        m0 = Matrix([[5.1,3.5,1.4,0.2],
                     [4.9,3.0,1.4,0.2],
                     [4.7,3.2,1.3,0.2],
                     [4.6,3.1,1.5,0.2],
                     [5.0,3.6,1.4,0.2],
                     [5.4,3.9,1.7,0.4]])
        m0.scale(0.5).shift(3*LEFT)
        t0 = Text("Setosa").shift(3*LEFT+1.4*UP).scale(0.5)
        elips0 = Text("...").shift(3*LEFT+1.5*DOWN).scale(0.5)
        m1 = Matrix([[7.0,3.2,4.7,1.4],
                     [6.4,3.2,4.5,1.5],
                     [6.9,3.1,4.9,1.5],
                     [5.5,2.3,4.0,1.3],
                     [6.5,2.8,4.6,1.5],
                     [5.7,2.8,4.5,1.3]])
        m1.scale(0.5)
        t1 = Text("Versicolor").shift(1.4*UP).scale(0.5)
        elips1 = Text("...").shift(1.5*DOWN).scale(0.5)
        m2 = Matrix([[6.3,3.3,6.0,2.5],
                     [5.8,2.7,5.1,1.9],
                     [7.1,3.0,5.9,2.1],
                     [6.3,2.9,5.6,1.8],
                     [6.5,3.0,5.8,2.2],
                     [7.6,3.0,6.6,2.1]])
        m2.scale(0.5).shift(3*RIGHT)
        t2 = Text("Virginica").shift(3*RIGHT+1.4*UP).scale(0.5)
        elips2 = Text("...").shift(3*RIGHT+1.5*DOWN).scale(0.5)
        self.play(FadeIn(m0, t0, elips0))
        self.play(FadeIn(m1, t1, elips1))
        self.play(FadeIn(m2, t2, elips2))
        self.wait(17)
        iris_data = VGroup(m0, t0, elips0, m1, t1, elips1, m2, t2, elips2)
        self.play(iris_data.animate.scale(0.5))
        self.play(iris_data.animate.shift(3*LEFT))

        squares_per_side_div_2 = 8
        resolution_fa = 1
        scale_factor = 0.3
        grid = []
        for x in range(squares_per_side_div_2-1, -squares_per_side_div_2-1, -1):
            for y in range(-squares_per_side_div_2, squares_per_side_div_2):
                s = Square().scale(scale_factor).shift(scale_factor*x*RIGHT+scale_factor*y*UP)
                s.set_stroke(width=1)
                grid.append(s)
        som3d = VGroup()
        for p in grid:
            som3d.add(p)
        som3d.shift(3*RIGHT+2*UP)
        som3d.rotate(angle=-PI/2, axis=RIGHT)
        som3d.rotate(angle=-PI/3, axis=UP)
        som3d.rotate(angle=PI/12, axis=RIGHT)
        sample = Tex("$[x_1, x_2, x_3, ..., x_n]$").shift(3*RIGHT-1*UP)
        w0 = Line(start=1.5*RIGHT-0.8*UP, end=1.8*RIGHT+1.45*UP)
        w1 = Line(start=2.2*RIGHT-0.8*UP, end=1.8*RIGHT+1.45*UP)
        w2 = Line(start=2.9*RIGHT-0.8*UP, end=1.8*RIGHT+1.45*UP)
        w3 = Line(start=4.2*RIGHT-0.8*UP, end=1.8*RIGHT+1.45*UP)
        self.play(FadeIn(som3d, sample, w0, w1, w2, w3))
        self.wait(1)

        e0 = Ellipse(width=1.5, height=0.16, color=WHITE).shift(1.5*LEFT-0.28*UP)
        a0 = CurvedArrow(start_point=1.5*LEFT-0.36*UP, end_point=3*RIGHT-1.3*UP)
        e0.set_stroke(width=1)
        a0.set_stroke(width=2)
        self.play(FadeIn(e0, a0))
        self.wait(18)
        self.play(FadeOut(iris_data, som3d, sample, w0, w1, w2, w3, e0, a0))
        self.wait(1)
        
        #iris_img = ImageMobject("iris_som.png").scale(1.6)
        #iris_overlay_img = ImageMobject("iris_som_overlay_edit.jpg").scale(1.6)
        #self.play(FadeIn(iris_img))
        #self.wait(1)
        #self.play(Transform(iris_img, iris_overlay_img))
        #self.wait(1)
        #self.play(FadeOut(iris_img))
        #self.wait(1)





class SomExplanation(Scene):
    def construct(self):
        h=0.13
        rects = []
        r0 = Rectangle(height=h, width=2, fill_opacity=1, color="#ff0000")
        rects.append(r0)
        r1 = Rectangle(height=h, width=2, fill_opacity=1, color="#ff4000")
        rects.append(r1)
        r2 = Rectangle(height=h, width=2, fill_opacity=1, color="#ff8000")
        rects.append(r2)
        r3 = Rectangle(height=h, width=2, fill_opacity=1, color="#ffc000")
        rects.append(r3)
        r4 = Rectangle(height=h, width=2, fill_opacity=1, color="#ffff00")
        rects.append(r4)
        r5 = Rectangle(height=h, width=2, fill_opacity=1, color="#c0ff00")
        rects.append(r5)
        r6 = Rectangle(height=h, width=2, fill_opacity=1, color="#80ff00")
        rects.append(r6)
        r7 = Rectangle(height=h, width=2, fill_opacity=1, color="#40ff00")
        rects.append(r7)
        r8 = Rectangle(height=h, width=2, fill_opacity=1, color="#00ff00")
        rects.append(r8)
        r9 = Rectangle(height=h, width=2, fill_opacity=1, color="#00ff40")
        rects.append(r9)
        r10 = Rectangle(height=h, width=2, fill_opacity=1, color="#00ff80")
        rects.append(r10)
        r11 = Rectangle(height=h, width=2, fill_opacity=1, color="#00ffc0")
        rects.append(r11)
        r12 = Rectangle(height=h, width=2, fill_opacity=1, color="#00ffff")
        rects.append(r12)
        r13 = Rectangle(height=h, width=2, fill_opacity=1, color="#00c0ff")
        rects.append(r13)
        r14 = Rectangle(height=h, width=2, fill_opacity=1, color="#0080ff")
        rects.append(r14)
        r15 = Rectangle(height=h, width=2, fill_opacity=1, color="#0040ff")
        rects.append(r15)
        r16 = Rectangle(height=h, width=2, fill_opacity=1, color="#0000ff")
        rects.append(r16)
        r17 = Rectangle(height=h, width=2, fill_opacity=1, color="#4000ff")
        rects.append(r17)
        r18 = Rectangle(height=h, width=2, fill_opacity=1, color="#8000ff")
        rects.append(r18)
        r19 = Rectangle(height=h, width=2, fill_opacity=1, color="#c000ff")
        rects.append(r19)
        r20 = Rectangle(height=h, width=2, fill_opacity=1, color="#ff00ff")
        rects.append(r20)
        for i, rect in enumerate(rects):
            rect.shift(((21-i)/6)*UP)
        gradient = VGroup(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20)
        gradient.move_to(ORIGIN)
        gradient.shift(4*LEFT)
        gradient.scale(0.5)
        #self.play(FadeIn(gradient))
        #self.wait(3)

        def param_plane(u, v):
            x = u
            y = v
            z = 0
            return np.array([x, y, z])

        # next: hard code in location of red, green, and blue pixels, and start by picking red, green, and blue for a couple iterations,
        #       then do some random iterations

        random.seed(1)
        squares_per_side_div_2 = 8
        resolution_fa = 1
        grid = []
        colors = []
        Wnew = np.zeros((3, (2*squares_per_side_div_2)**2))
        for x in range(squares_per_side_div_2-1, -squares_per_side_div_2-1, -1):
            for y in range(-squares_per_side_div_2, squares_per_side_div_2):
                row = (squares_per_side_div_2 - 1) - x
                col = squares_per_side_div_2 + y
                if x==3 and y==3:
                    c="#0000ff"
                    colors.append(np.asarray([0,0,1]))
                    Wnew[:, 2*squares_per_side_div_2*row + col] = np.asarray([0,0,1])
                elif x==-3 and y==3:
                    c="#ff0000"
                    colors.append(np.asarray([1,0,0]))
                    Wnew[:, 2*squares_per_side_div_2*row + col] = np.asarray([1,0,0])
                elif x==-3 and y==-3:
                    c="#00ff00"
                    colors.append(np.asarray([0,1,0]))
                    Wnew[:, 2*squares_per_side_div_2*row + col] = np.asarray([0,1,0])
                else:
                    c = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
                    c = c / np.linalg.norm(c)
                    colors.append(c)
                    Wnew[:, (2*squares_per_side_div_2*row + col)] = c
                    c = np.round(c*255)
                    r = hex(int(c[0]))[2:]
                    if len(r) == 1:
                        r = "0" + r
                    g = hex(int(c[1]))[2:]
                    if len(g) == 1:
                        g = "0" + g
                    b = hex(int(c[2]))[2:]
                    if len(b) == 1:
                        b = "0" + b
                    c = "#" + r + g + b
                # c = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
                # c = c / np.linalg.norm(c)
                # colors.append(c)
                # Wnew[:, (2*squares_per_side_div_2*row + col)] = c
                # c = np.round(c*255)
                # r = hex(int(c[0]))[2:]
                # if len(r) == 1:
                #     r = "0" + r
                # g = hex(int(c[1]))[2:]
                # if len(g) == 1:
                #     g = "0" + g
                # b = hex(int(c[2]))[2:]
                # if len(b) == 1:
                #     b = "0" + b
                # c = "#" + r + g + b
                p = ParametricSurface(
                    param_plane,
                    resolution=(resolution_fa, resolution_fa),
                    u_range=[x,x+1],
                    v_range=[y,y+1]
                    )
                p.set_style(fill_color=c, fill_opacity=1)
                grid.append(p)
        som3d = VGroup()
        for p in grid:
            som3d.add(p)
        som3d.scale_about_point(0.4, ORIGIN)
        som3d.shift(1.85*RIGHT)
        #self.play(FadeIn(som3d))
        #self.wait(1)

        # 1=blue, 5=green, 16=red

        X = get_som_input(3)
        (N, dim) = X.shape
        sigma = 2*squares_per_side_div_2/2
        eta = 0.99
        alpha = 0.99
        Wold = np.zeros((3, (2*squares_per_side_div_2)**2))
        pos = np.zeros(((2*squares_per_side_div_2), (2*squares_per_side_div_2), 2))
        for i in range((2*squares_per_side_div_2)):
            for j in range((2*squares_per_side_div_2)):
                pos[i,j] = [i, j]
        i = 0
        while i < 3:
            Wold = np.copy(Wnew)
            if i==0:
                n=1
            elif i==1:
                n=5
            elif i==2:
                n=16
            else:
                n = random.randint(0, N-1)
            xtrain = X[n,:] / np.linalg.norm(X[n,:])
            a = (np.matmul(xtrain, Wold)).reshape((2*squares_per_side_div_2, 2*squares_per_side_div_2))
            [max_row, max_col] = np.unravel_index(np.argmax(a), a.shape)
            d_arr = np.linalg.norm((pos - np.asarray([max_row, max_col])), axis=2)
            u_arr = np.exp((-1) * (d_arr**2) / (2*(sigma**2)))
            Wnew = Wold + (eta*(u_arr.flatten())) * (xtrain.reshape(3,1) - Wold)
            Wnew = Wnew / np.linalg.norm(Wnew, axis=0)
            eta = eta*alpha
            sigma = sigma*alpha
            i += 1
            for c in range((2*squares_per_side_div_2)**2):
                c_new = Wnew[:,c]
                p = grid[c]
                c_new = np.round(c_new*255)
                r = hex(int(c_new[0]))[2:]
                if len(r) == 1:
                    r = "0" + r
                g = hex(int(c_new[1]))[2:]
                if len(g) == 1:
                    g = "0" + g
                b = hex(int(c_new[2]))[2:]
                if len(b) == 1:
                    b = "0" + b
                c_new = "#" + r + g + b
                p.set_style(fill_color=c_new, fill_opacity=1)
            #self.wait(1)
        self.play(FadeIn(gradient, som3d))
        self.wait(3)

        a2 = Arrow(start=LEFT, end=RIGHT).shift(5*LEFT + -0.5*UP).scale(0.5)
        t2 = Tex("Sample A").shift(6*LEFT + -0.5*UP).scale(0.5)
        s2 = Circle(radius=0.25, color=WHITE).shift(4.85*RIGHT+2.62*UP)
        self.play(FadeIn(s2))
        self.play(FadeIn(a2, t2))
        self.wait(1)

        a0 = Arrow(start=LEFT, end=RIGHT).shift(5*LEFT + 0.85*UP).scale(0.5)
        t0 = Tex("Sample B").shift(6*LEFT + 0.85*UP).scale(0.5)
        s0 = Circle(radius=0.25, color=WHITE).shift(-0.35*RIGHT+1.8*UP)
        self.play(FadeIn(s0))
        self.play(FadeIn(a0, t0))
        self.wait(1)

        a1 = Arrow(start=LEFT, end=RIGHT).shift(5*LEFT + 0.15*UP).scale(0.5)
        t1 = Tex("Sample C").shift(6*LEFT + 0.15*UP).scale(0.5)
        s1 = Circle(radius=0.25, color=WHITE).shift(2.05*RIGHT+-2.61*UP)
        self.play(FadeIn(s1))
        self.play(FadeIn(a1, t1))
        self.wait(6)

        e0 = Ellipse(width=2.3, height=0.2, color=WHITE).shift(4*LEFT + -0.5*UP).scale(0.5)
        self.play(FadeIn(e0))
        self.wait(6)

        Wold = np.copy(Wnew)
        n = 1
        xtrain = X[n,:] / np.linalg.norm(X[n,:])
        a = (np.matmul(xtrain, Wold)).reshape((2*squares_per_side_div_2, 2*squares_per_side_div_2))
        [max_row, max_col] = np.unravel_index(np.argmax(a), a.shape)
        d_arr = np.linalg.norm((pos - np.asarray([max_row, max_col])), axis=2)
        u_arr = np.exp((-1) * (d_arr**2) / (2*(sigma**2)))
        Wnew = Wold + (eta*(u_arr.flatten())) * (xtrain.reshape(3,1) - Wold)
        Wnew = Wnew / np.linalg.norm(Wnew, axis=0)
        eta = eta*alpha
        sigma = sigma*alpha
        for c in range((2*squares_per_side_div_2)**2):
            c_new = Wnew[:,c]
            p = grid[c]
            c_new = np.round(c_new*255)
            r = hex(int(c_new[0]))[2:]
            if len(r) == 1:
                r = "0" + r
            g = hex(int(c_new[1]))[2:]
            if len(g) == 1:
                g = "0" + g
            b = hex(int(c_new[2]))[2:]
            if len(b) == 1:
                b = "0" + b
            c_new = "#" + r + g + b
            p.set_style(fill_color=c_new, fill_opacity=1)
        self.wait(4)
        
        s3 = Circle(radius=0.25, color=WHITE).shift(-1.15*RIGHT+0.2*UP)
        a3 = Arrow(start=-0.35*RIGHT+1.8*UP, end=-1.15*RIGHT+0.2*UP)
        self.play(FadeIn(s3, a3))
        #self.wait(1)

        s4 = Circle(radius=0.25, color=WHITE).shift(1.25*RIGHT+-3.0*UP)
        a4 = Arrow(start=2.05*RIGHT+-2.61*UP, end=1.25*RIGHT+-3.0*UP)
        self.play(FadeIn(s4, a4))
        self.wait(40)

        s5 = Circle(radius=6.6, color=WHITE).shift(4.85*RIGHT+2.62*UP)
        s5_dashed = DashedVMobject(s5)
        self.play(FadeIn(s5_dashed))
        self.wait(5)






class SomAlgorithm(ThreeDScene):
    def construct(self):
        generic_training_sample = Tex("$\\vec{x_i} = [x_1, ..., x_n]$")
        self.play(FadeIn(generic_training_sample))
        self.wait(3.5)
        training_sample = Tex("$\\vec{x_i} = [x_1, x_2, x_3]$")
        self.play(Transform(generic_training_sample, training_sample))
        self.wait(3)
        start = Tex("$\\vec{x_i} = [$")
        r = Tex("$x_1$", color=RED).shift(RIGHT)
        c1 = Tex("$,$").shift(1.3*RIGHT)
        g = Tex("$x_2$", color=GREEN).shift(1.7*RIGHT)
        c2 = Tex("$,$").shift(2*RIGHT)
        b = Tex("$x_3$", color=BLUE).shift(2.4*RIGHT)
        bracket = Tex("$]$").shift(2.7*RIGHT)
        colored_training_sample = VGroup(start, r, c1, g, c2, b, bracket)
        colored_training_sample.move_to(ORIGIN)
        self.play(Transform(generic_training_sample, colored_training_sample))
        self.wait(5)
        self.play(FadeOut(generic_training_sample))
        self.wait(1)

        m0 = Matrix([[255, 0, 0],
             [255, 127, 0],
             [255, 255, 0],
             [127, 255, 0],
             [0, 255, 0],
             [0, 255, 127],
             [0, 255, 255],
             [0, 127, 255],
             [0, 0, 255],
             [127, 0, 255],
             [255, 0, 255]])
        m0.scale(0.4)
        self.play(FadeIn(m0))
        self.wait(4)
        self.play(m0.animate.shift(2*LEFT))

        a0 = Arrow(start=LEFT, end=RIGHT)
        self.play(FadeIn(a0))

        h=0.13
        rects = []
        r0 = Rectangle(height=h, width=2, fill_opacity=1, color="#ff0000")
        rects.append(r0)
        r1 = Rectangle(height=h, width=2, fill_opacity=1, color="#ff4000")
        rects.append(r1)
        r2 = Rectangle(height=h, width=2, fill_opacity=1, color="#ff8000")
        rects.append(r2)
        r3 = Rectangle(height=h, width=2, fill_opacity=1, color="#ffc000")
        rects.append(r3)
        r4 = Rectangle(height=h, width=2, fill_opacity=1, color="#ffff00")
        rects.append(r4)
        r5 = Rectangle(height=h, width=2, fill_opacity=1, color="#c0ff00")
        rects.append(r5)
        r6 = Rectangle(height=h, width=2, fill_opacity=1, color="#80ff00")
        rects.append(r6)
        r7 = Rectangle(height=h, width=2, fill_opacity=1, color="#40ff00")
        rects.append(r7)
        r8 = Rectangle(height=h, width=2, fill_opacity=1, color="#00ff00")
        rects.append(r8)
        r9 = Rectangle(height=h, width=2, fill_opacity=1, color="#00ff40")
        rects.append(r9)
        r10 = Rectangle(height=h, width=2, fill_opacity=1, color="#00ff80")
        rects.append(r10)
        r11 = Rectangle(height=h, width=2, fill_opacity=1, color="#00ffc0")
        rects.append(r11)
        r12 = Rectangle(height=h, width=2, fill_opacity=1, color="#00ffff")
        rects.append(r12)
        r13 = Rectangle(height=h, width=2, fill_opacity=1, color="#00c0ff")
        rects.append(r13)
        r14 = Rectangle(height=h, width=2, fill_opacity=1, color="#0080ff")
        rects.append(r14)
        r15 = Rectangle(height=h, width=2, fill_opacity=1, color="#0040ff")
        rects.append(r15)
        r16 = Rectangle(height=h, width=2, fill_opacity=1, color="#0000ff")
        rects.append(r16)
        r17 = Rectangle(height=h, width=2, fill_opacity=1, color="#4000ff")
        rects.append(r17)
        r18 = Rectangle(height=h, width=2, fill_opacity=1, color="#8000ff")
        rects.append(r18)
        r19 = Rectangle(height=h, width=2, fill_opacity=1, color="#c000ff")
        rects.append(r19)
        r20 = Rectangle(height=h, width=2, fill_opacity=1, color="#ff00ff")
        rects.append(r20)
        for i, rect in enumerate(rects):
            rect.shift(((21-i)/6)*UP)
        gradient = VGroup(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20)
        gradient.move_to(ORIGIN)
        gradient.shift(2*RIGHT)
        self.play(FadeIn(gradient))
        self.wait(3)
        som_input = VGroup(gradient, a0, m0)
        self.play(som_input.animate.scale(0.5))
        self.play(som_input.animate.shift(4.5*LEFT))
        self.wait(3)

        def param_plane(u, v):
            x = u
            y = v
            z = 0
            return np.array([x, y, z])

        squares_per_side_div_2 = 8
        resolution_fa = 1
        grid = []
        colors = []
        Wnew = np.zeros((3, (2*squares_per_side_div_2)**2))
        for x in range(squares_per_side_div_2-1, -squares_per_side_div_2-1, -1):
        #for x in range(-squares_per_side_div_2, squares_per_side_div_2):
            for y in range(-squares_per_side_div_2, squares_per_side_div_2):
                row = (squares_per_side_div_2 - 1) - x
                col = squares_per_side_div_2 + y
                if x==0 and y==2:
                    c="#0000ff"
                    colors.append(np.asarray([0,0,1]))
                    Wnew[:, 2*squares_per_side_div_2*row + col] = np.asarray([0,0,1])
                else:
                    c = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
                    c = c / np.linalg.norm(c)
                    colors.append(c)
                    Wnew[:, (2*squares_per_side_div_2*row + col)] = c
                    c = np.round(c*255)
                    r = hex(int(c[0]))[2:]
                    if len(r) == 1:
                        r = "0" + r
                    g = hex(int(c[1]))[2:]
                    if len(g) == 1:
                        g = "0" + g
                    b = hex(int(c[2]))[2:]
                    if len(b) == 1:
                        b = "0" + b
                    c = "#" + r + g + b
                p = ParametricSurface(
                    param_plane,
                    resolution=(resolution_fa, resolution_fa),
                    u_range=[x,x+1],
                    v_range=[y,y+1]
                    )
                p.set_style(fill_color=c, fill_opacity=1)
                grid.append(p)
        som3d = VGroup()
        for p in grid:
            som3d.add(p)
        som3d.scale_about_point(0.4, ORIGIN)
        som3d.shift(1.85*RIGHT)
        self.play(FadeIn(som3d))
        self.wait(1)

        s0 = Circle(radius=0.25, color=WHITE).shift(1.16*LEFT+3*UP)
        self.play(FadeIn(s0))
        self.wait(1)

        a1 = Arrow(start=RIGHT, end=LEFT).shift(2.2*LEFT + 3*UP)
        self.play(FadeIn(a1))

        generic_weight_vector = Tex("$\\vec{w_i} = [w_1, ..., w_n]$").shift(4.8*LEFT + 3*UP)
        self.play(FadeIn(generic_weight_vector))
        self.wait(12)
        weight_vector = Tex("$\\vec{w_i} = [w_1, w_2, w_3]$").shift(4.8*LEFT + 3*UP)
        self.play(Transform(generic_weight_vector, weight_vector))
        self.wait(8)

        self.play(FadeOut(s0), FadeOut(a1), FadeOut(generic_weight_vector))
        self.wait(1)

        self.play(
           Rotate(
               som3d,
               -PI/2,
               run_time=1,
               axis=RIGHT
           )
        )
        self.play(
           Rotate(
               som3d,
               -PI/3,
               run_time=1,
               axis=UP
           )
        )
        self.play(
           Rotate(
               som3d,
               PI/12,
               run_time=2,
               axis=RIGHT
           )
        )
        self.wait(1)

        axes = ThreeDAxes()
        axes.rotate(angle=-PI/3, axis=UP)
        axes.rotate(angle=PI/12, axis=RIGHT)
        axes.shift(1.85*RIGHT)
        self.play(FadeIn(axes))
        self.wait(3)

        e0 = Ellipse(width=3.5, height=0.2, color=WHITE)
        e0.shift(4.5*LEFT + -0.5*UP)
        self.play(FadeIn(e0))
        self.wait(2)

        activation = Tex("$a_i = \\vec{w_i} \\cdot \\vec{x} $").shift(2*LEFT + 2*UP)
        self.play(FadeIn(activation))
        self.wait(3)

        s1 = Circle(radius=0.25, color=WHITE)
        s1.shift(2.78*RIGHT+0.13*UP)
        self.play(FadeIn(s1))
        self.wait(6)

        self.play(FadeOut(activation))

        def param_gauss(u, v):
           x = u
           y = v
           d = np.sqrt(x * x + y * y)
           sigma, mu = 0.2, 0.0
           z = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
           return np.array([x, y, z])

        gauss_plane = ParametricSurface(
           param_gauss,
           resolution=(22, 22),
           v_range=[-2,2],
           u_range=[-2,2]
        )
        gauss_plane.rotate(angle=-PI/2, axis=RIGHT)
        gauss_plane.rotate(angle=-PI/3, axis=UP)
        gauss_plane.rotate(angle=PI/12, axis=RIGHT)

        gauss_plane.shift(1.4*RIGHT + 0.6*UP)
        gauss_plane.scale_about_point(1.9, ORIGIN)
        gauss_plane.set_style(fill_opacity=1)
        gauss_plane.set_style(stroke_color=GREEN)
        gauss_plane.set_fill_by_checkerboard(GREEN, BLUE, opacity=0.1)

        update = Tex("$\\vec{w^{new}_i} = \\vec{w^{old}_i} + e^{-d^2/\\sigma^2} \\cdot \\eta \\cdot (\\vec{x} - \\vec{w^{old}_i})$").shift(2.4*LEFT + 2*UP)
        self.play(FadeIn(gauss_plane, update))
        self.wait(12)

        s2 = Circle(radius=0.25, color=WHITE)
        s2.shift(0.78*RIGHT+0*UP)
        self.play(FadeIn(s2))
        self.wait(1)
        
        a4 = Arrow(start=2.78*RIGHT+0.13*UP, end=0.78*RIGHT)
        self.play(FadeIn(a4))
        self.wait(3)

        self.play(FadeOut(s2, a4))
        self.wait(1)

        l0 = Line(start=2.15*RIGHT+0.82*UP, end=3.42*RIGHT+0.8*UP)
        self.play(FadeIn(l0))
        self.wait(4)

        self.play(FadeOut(l0))
        self.wait(8)

        x=-8
        y=-8
        sigma = 3
        eta = 0.9
        i = 0
        for x in range(squares_per_side_div_2-1, -squares_per_side_div_2-1, -1):
        #for x in range(-squares_per_side_div_2, squares_per_side_div_2):
            for y in range(-squares_per_side_div_2, squares_per_side_div_2):
                c = colors[i]
                p = grid[i]
                row = (squares_per_side_div_2 - 1) - x
                col = squares_per_side_div_2 + y
                d = np.linalg.norm(np.asarray([x, y]) - np.asarray([0, 2]))
                u = np.exp((-1) * (d**2) / (2*(sigma**2)))
                c_new = c + eta*u*(np.asarray([0, 0, 1]) - c)
                c_new = c_new / np.linalg.norm(c_new)
                colors[i] = c_new
                Wnew[:, 2*squares_per_side_div_2*row + col] = c_new
                c_new = np.round(c_new*255)
                r = hex(int(c_new[0]))[2:]
                if len(r) == 1:
                    r = "0" + r
                g = hex(int(c_new[1]))[2:]
                if len(g) == 1:
                    g = "0" + g
                b = hex(int(c_new[2]))[2:]
                if len(b) == 1:
                    b = "0" + b
                c_new = "#" + r + g + b
                p.set_style(fill_color=c_new, fill_opacity=1)
                i += 1
        self.wait(8)

        self.play(FadeOut(e0), FadeOut(s1), FadeOut(update), FadeOut(gauss_plane))
        self.wait(1)

        eta_obj = Tex("$\\eta$").shift(2*LEFT + 2*UP)
        sigma_obj = Tex("$\\sigma$").shift(2.9*LEFT + 2*UP)
        a2 = Arrow(start=UP, end=DOWN).shift(1.7*LEFT + 2*UP)
        a3 = Arrow(start=UP, end=DOWN).shift(2.6*LEFT + 2*UP)
        self.play(FadeIn(eta_obj), FadeIn(sigma_obj), FadeIn(a2), FadeIn(a3))
        self.wait(5)

        X = get_som_input(3)
        (N, dim) = X.shape
        sigma = 3
        eta = 0.9
        alpha = 0.98
        Wold = np.zeros((3, (2*squares_per_side_div_2)**2))
        pos = np.zeros(((2*squares_per_side_div_2), (2*squares_per_side_div_2), 2))
        for i in range((2*squares_per_side_div_2)):
            for j in range((2*squares_per_side_div_2)):
                pos[i,j] = [i, j]
        i = 0
        while i < 50:
            Wold = np.copy(Wnew)
            n = random.randint(0, N-1)
            xtrain = X[n,:] / np.linalg.norm(X[n,:])
            a = (np.matmul(xtrain, Wold)).reshape((2*squares_per_side_div_2, 2*squares_per_side_div_2))
            [max_row, max_col] = np.unravel_index(np.argmax(a), a.shape)
            d_arr = np.linalg.norm((pos - np.asarray([max_row, max_col])), axis=2)
            u_arr = np.exp((-1) * (d_arr**2) / (2*(sigma**2)))
            Wnew = Wold + (eta*(u_arr.flatten())) * (xtrain.reshape(3,1) - Wold)
            Wnew = Wnew / np.linalg.norm(Wnew, axis=0)
            eta = eta*alpha
            sigma = sigma*alpha
            i += 1
            for c in range((2*squares_per_side_div_2)**2):
                c_new = Wnew[:,c]
                p = grid[c]
                c_new = np.round(c_new*255)
                r = hex(int(c_new[0]))[2:]
                if len(r) == 1:
                    r = "0" + r
                g = hex(int(c_new[1]))[2:]
                if len(g) == 1:
                    g = "0" + g
                b = hex(int(c_new[2]))[2:]
                if len(b) == 1:
                    b = "0" + b
                c_new = "#" + r + g + b
                p.set_style(fill_color=c_new, fill_opacity=1)
            self.wait(0.5)

        self.play(FadeOut(eta_obj), FadeOut(sigma_obj), FadeOut(a2), FadeOut(a3))
        self.wait(1)

        a4 = Arrow(start=RIGHT, end=LEFT).shift(2.5*LEFT + 0.8*UP).scale(0.5)
        self.play(FadeIn(a4))
        self.wait(1)

        self.play(FadeIn(activation))
        self.wait(3)

        winningNodes = set()
        for i in range(N):
            xtrain = X[i,:] / np.linalg.norm(X[i,:])
            a = (np.matmul(xtrain, Wnew)).reshape(((2*squares_per_side_div_2), (2*squares_per_side_div_2)))
            [row, col] = np.unravel_index(np.argmax(a), a.shape)
            winningNodes.add((row, col))
        for c in range((2*squares_per_side_div_2)**2):
            p = grid[c]
            row = c//(2*squares_per_side_div_2)
            col = c%(2*squares_per_side_div_2)
            if (row, col) not in winningNodes:
                p.set_style(fill_color="#000000", fill_opacity=1)
            if c%(2*squares_per_side_div_2) == 0:
                self.wait(0.5)
        self.wait(3)



class SomAnimation3(Scene):
  def construct(self):
    shift_amt = 4.8
    scale_amt = 1.2
    for i in range(0,439):
      if i == 0:
        self.play(FadeIn(ImageMobject("animation2/winners_16_100_3_"+str(i)+".png").scale(scale_amt).shift(0.0*LEFT)), FadeIn(ImageMobject("animation2/colors_16_100_3_"+str(i)+".png").scale(scale_amt).shift(shift_amt*LEFT)))
      elif i == 1:
        self.play(FadeOut(ImageMobject("animation2/winners_16_100_3_"+str(i-1)+".png").scale(scale_amt).shift(0.0*LEFT)), FadeOut(ImageMobject("animation2/colors_16_100_3_"+str(i-1)+".png").scale(scale_amt).shift(shift_amt*LEFT)), FadeIn(ImageMobject("animation2/winners_16_100_3_"+str(i)+".png").scale(scale_amt).shift(0.0*LEFT)), FadeIn(ImageMobject("animation2/changes_16_100_3_"+str(i)+".png").scale(scale_amt).shift(shift_amt*RIGHT)), FadeIn(ImageMobject("animation2/colors_16_100_3_"+str(i)+".png").scale(scale_amt).shift(shift_amt*LEFT)))
      else:
        self.play(FadeOut(ImageMobject("animation2/winners_16_100_3_"+str(i-1)+".png").scale(scale_amt).shift(0.0*LEFT)), FadeOut(ImageMobject("animation2/changes_16_100_3_"+str(i-1)+".png").scale(scale_amt).shift(shift_amt*RIGHT)), FadeOut(ImageMobject("animation2/colors_16_100_3_"+str(i-1)+".png").scale(scale_amt).shift(shift_amt*LEFT)), FadeIn(ImageMobject("animation2/winners_16_100_3_"+str(i)+".png").scale(scale_amt).shift(0.0*LEFT)), FadeIn(ImageMobject("animation2/changes_16_100_3_"+str(i)+".png").scale(scale_amt).shift(shift_amt*RIGHT)), FadeIn(ImageMobject("animation2/colors_16_100_3_"+str(i)+".png").scale(scale_amt).shift(shift_amt*LEFT)))

