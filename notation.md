\begin{bmatrix}
\dfrac{M}{dt} + \left(b + c + \dfrac{\partial c}{\partial \dot{q}} \cdot \dot{q}\right) & -k_t \cdot gr \\[1em]
k_e \cdot gr & \dfrac{L}{dt} + R
\end{bmatrix}
\begin{bmatrix}
\Delta \dot{q} \\[0.5em]
\Delta i
\end{bmatrix}
=
\begin{bmatrix}
k_t \cdot gr \cdot i_k - b\dot{q} - c\dot{q} - G + \tau_{applied} + \tau_{constraint} \\[0.5em]
V - R i_k - k_e \cdot gr \cdot \dot{q}_k
\end{bmatrix}

각각 좌변 항을 A, B, C, D 행렬, 우변 항을 F, G 행렬이라고 했을때 schur=-BD^-1C로 작성했었고, RHS 보정 항=-BD^-1G로 작성했었어. 즉 delta qdot=[A+schur]^-1 * [F+RHS 보정] 이렇게 계산했었어. 지금은 M_eff 같은 notation으로 작성해서 이부분이 헷갈림. 또, schur의 부호 주의해주고, force 항에 댐핑(b) 빠졌어.