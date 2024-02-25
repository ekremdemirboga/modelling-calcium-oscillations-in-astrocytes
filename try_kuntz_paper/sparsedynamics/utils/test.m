function dy = test(t,y,sigma,beta,rho)



dy = [
    v_in - k_out*y(1) + V_CICR- v_serca + k_f*(y(2) - y(1));
    v_serca - v_CICR - k_f*(y(2) - y(1));
    v_PLC - k_deg*y(3)
]