module pml

  implicit none

contains

  subroutine step(f1, f2, phi1, phi2, sigma,                           &
      model_padded, dt, dx, sources, sources_x, num_steps, pml_width,  &
      pad_width)

    real, intent (in out), dimension (:) :: f1
    real, intent (in out), dimension (:) :: f2
    real, intent (in out), dimension (:) :: phi1
    real, intent (in out), dimension (:) :: phi2
    real, intent (in), dimension (:) :: sigma
    real, intent (in), dimension (:) :: model_padded
    real, intent (in) :: dt
    real, intent (in) :: dx
    real, intent (in), dimension (:, :) :: sources
    integer, intent (in), dimension (:) :: sources_x
    integer, intent (in) :: num_steps
    integer, intent (in) :: pml_width
    integer, intent (in) :: pad_width

    integer :: step_idx
    logical :: even

    do step_idx = 1, num_steps
    even = (mod (step_idx, 2) == 0)
    if (even) then
      call one_step(f2, f1, phi2, phi1, sigma,                         &
        model_padded, dt, dx, sources, sources_x, step_idx, pml_width, &
        pad_width)
    else
      call one_step(f1, f2, phi1, phi2, sigma,                         &
        model_padded, dt, dx, sources, sources_x, step_idx, pml_width, &
        pad_width)
    end if
    end do

  end subroutine step


  subroutine one_step(f, fp, phi, phip, sigma,                         &
      model_padded, dt, dx, sources, sources_x, step_idx, pml_width,   &
      pad_width)

    real, intent (in out), dimension (:) :: f
    real, intent (in out), dimension (:) :: fp
    real, intent (in), dimension (:) :: phi
    real, intent (in out), dimension (:) :: phip
    real, intent (in), dimension (:) :: sigma
    real, intent (in), dimension (:) :: model_padded
    real, intent (in) :: dt
    real, intent (in) :: dx
    real, intent (in), dimension (:, :)  :: sources
    integer, intent (in), dimension (:) :: sources_x
    integer, intent (in) :: step_idx
    integer, intent (in) :: pml_width
    integer, intent (in) :: pad_width

    integer :: i
    integer :: nx_padded
    integer :: num_sources

    nx_padded = size(f)
    num_sources = size(sources, dim=1)

    ! Propagate
    do i = pad_width + 1, nx_padded - pad_width
    call fd_pml(f, fp, phi, phip, sigma, model_padded, dt, dx, i)
    end do

    ! source term
    do i = 1, num_sources
    call add_source(fp, model_padded, dt, sources(i, step_idx),        &
      sources_x(i), pml_width + pad_width)
    end do

  end subroutine one_step

  
  subroutine fd_pml(f, fp, phi, phip, sigma, model_padded, dt, dx, i)

    real, intent (in), dimension (:) :: f
    real, intent (in out), dimension (:) :: fp
    real, intent (in), dimension (:) :: phi
    real, intent (in out), dimension (:) :: phip
    real, intent (in), dimension (:) :: sigma
    real, intent (in), dimension (:) :: model_padded
    real, intent (in) :: dt
    real, intent (in) :: dx
    integer, intent (in) :: i

    ! Based on PML explanation by Steven G. Johnson
    ! http://math.mit.edu/~stevenj/18.369/pml.pdf
    !
    ! Scalar wave equation: u_xx = u_tt/c^2  (1)
    !
    ! Split (1) into:
    ! phi_t = u_x  (2a)
    ! u_t = c^2 * phi_x  (2b)
    !
    ! Verification:
    ! x partial derivative of (2a): phi_(x,t) = u_xx  (3a)
    ! t partial derivative of (2b): u_tt = c^2 * phi_(x,t)  (3b)
    ! substitute (3a) in (3b): u_tt = c^2 * u_xx, which is (1).
    !
    ! For the PML region, we use the wavefield at
    ! x' = x + i * f(x) / omega, as this will exponentially
    ! damp the wavefield. Now, with sigma = f_x:
    ! u_x -> u_x / (1 + i * sigma / omega). Applying this to (2):
    ! phi_t = u_x / (1 + i * sigma / omega)  (4a)
    ! u_t = c^2 * phi_x / (1 + i * sigma / omega)  (4b).
    !
    ! Assuming that u and phi can be written in the form
    ! A * exp(i(kx - omega*t)), u_t = -i*omega*u. Applying
    ! this to (4) and multiplying both sides by the denominator
    ! of the right hand sides:
    ! (1 + i * sigma / omega) * (-i*omega) * phi = u_x  (5a)
    ! (1 + i * sigma / omega) * (-i*omega) * u = c^2 * phi_x  (5a)
    !
    ! Multiplying,
    ! (1 + i * sigma / omega) * (-i*omega) = -i*omega + sigma.
    ! Using this in (5), and noting that -i*omega*u = u_t,
    ! phi_t = u_x - sigma * phi  (6a)
    ! u_t = c^2 * phi_x - sigma * u  (6b)
    !
    ! Taking the x partial derivative of (6a), the t partial derivative
    ! of (6b), and substituting, as in (3), we get
    ! u_tt = c^2 * (u_xx - (sigma * phi)_x) - sigma * u_t (7)
    ! Expanding the derivative (sigma * phi)_x using the product rule:
    ! u_tt = c^2 * (u_xx - (sigma_x * phi + sigma * phi_x))
    !        - sigma * u_t (8)
    !
    ! Using central differences for u_tt and u_t:
    ! (u(t+1) - 2*u(t) + u(t-1))/dt^2 = c^2 * (u_xx - (sigma_x * phi
    !                                                  + sigma * phi_x))
    !                                   - sigma * (u(t+1) - u(t-1))/(2*dt)
    ! Rearranging:
    ! u(t+1) = c^2 * dt^2 / (1 + dt * sigma / 2)
    !          * (u_xx - (sigma_x * phi + sigma * phi_x))
    !          + dt * sigma / (2 + dt * sigma) * u(t-1)
    !          + 1 / (1 + dt * sigma / 2) * (2 * u(t) - u(t-1))  (9)
    !
    ! Using a forward difference for phi_t in (6a):
    ! (phi(t+1) - phi(t))/dt = u_x - sigma * phi(t),
    ! Rearranging:
    ! phi(t+1) = dt * u_x + phi(t) * (1 - dt * sigma)  (10)
    ! 
    ! 
    ! (9) shows how to update u, and (10) shows how to update phi.

    real :: f_xx
    real :: f_x
    real :: phi_x
    real :: sigma_x

    f_xx = second_x_deriv(f, i, dx)
    f_x = first_x_deriv(f, i, dx)
    phi_x = first_x_deriv(phi, i, dx)
    sigma_x = first_x_deriv(sigma, i, dx)

    ! (9)
    fp(i) = model_padded(i)**2 * dt**2 / (1 + dt * sigma(i)/2)         &
            * (f_xx - (sigma_x * phi(i) + sigma(i) * phi_x))           &
            + dt * sigma(i) * fp(i) / (2 + dt * sigma(i))              &
            + 1 / (1 + dt * sigma(i) / 2) * (2 * f(i) - fp(i))

    ! (10)
    phip(i) =  dt * f_x + phi(i) - dt * sigma(i)*phi(i)

  end subroutine fd_pml


  subroutine add_source(fp, model_padded, dt, source, source_x,        &
      total_pad)

    real, intent (in out), dimension (:) :: fp
    real, intent (in), dimension (:) :: model_padded
    real, intent (in) :: dt
    real, intent (in)  :: source
    integer, intent (in) :: source_x
    integer, intent (in) :: total_pad

    integer :: sx

    sx = source_x + total_pad + 1;

    fp(sx) = fp(sx) + (model_padded(sx)**2 * dt**2 * source)

  end subroutine add_source


  pure function first_x_deriv(f, i, dx)

    real, intent (in), dimension (:) :: f
    integer, intent (in) :: i
    real, intent (in) :: dx

    real :: first_x_deriv

    first_x_deriv = (                                                 &
      5*f(i-6)-72*f(i-5)                                              &
      +495*f(i-4)-2200*f(i-3)                                         &
      +7425*f(i-2)-23760*f(i-1)                                       &
      +23760*f(i+1)-7425*f(i+2)                                       &
      +2200*f(i+3)-495*f(i+4)                                         &
      +72*f(i+5)-5*f(i+6))/(27720*dx)

  end function first_x_deriv


  pure function second_x_deriv(f, i, dx)

    real, intent (in), dimension (:) :: f
    integer, intent (in) :: i
    real, intent (in) :: dx

    real :: second_x_deriv

    second_x_deriv = (                                                 &
      -735*f(i-8)+15360*f(i-7)                                         &
      -156800*f(i-6)+1053696*f(i-5)                                    & 
      -5350800*f(i-4)+22830080*f(i-3)                                  & 
      -94174080*f(i-2)+538137600*f(i-1)                                & 
      -924708642*f(i+0)                                                & 
      +538137600*f(i+1)-94174080*f(i+2)                                & 
      +22830080*f(i+3)-5350800*f(i+4)                                  & 
      +1053696*f(i+5)-156800*f(i+6)                                    & 
      +15360*f(i+7)-735*f(i+8))/(302702400*dx**2)

  end function second_x_deriv

end module pml
